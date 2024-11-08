import os
import json
import random
import safetensors.torch

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoProcessor,
    AutoModelForCausalLM,
)

from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

from datasets import load_dataset
from accelerate import Accelerator
# import torch
from torch.utils.data import Dataset
# import os
import torch
from aim.hugging_face import AimCallback
from transformers import TrainerCallback

NUM_PARAM = "8B"
DATA_PATH = "./data/ai_responses.json"

model_name = f"meta-llama/Llama-3.1-{NUM_PARAM}"  # 또는 다른 Llama-2 모델
# tokenizer = AutoProcessor.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
#    device_map="auto"
)

# 토크나이저 설정
processor.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(processor))




processor.padding_side = "right"

# 커스텀 데이터셋 클래스 정의
class TextDataset(Dataset):
    def __init__(self, texts, processor, max_length=512):
        self.processor = processor
        self.texts = texts
        self.max_length = max_length

    def __getitem__(self, idx):
        text = self.texts[idx]
        # processor를 사용하여 텍스트 처리
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        # squeeze를 사용하여 배치 차원 제거
        return {
            key: val.squeeze(0) for key, val in inputs.items()
        }

    def __len__(self):
        return len(self.texts)
    

def load_data(data_path):
    data_list = []
    with open(data_path, 'r') as file:
        data = json.load(file)
        for item in data:
            js = json.loads(item)
            photo_a_description = js['image_descriptions'][0]
            photo_b_description = js['image_descriptions'][1]
            conversations = js['conversation']
            problem = js['problem']
            solution = js['solution']
            choices = ".\n".join(js['choices']) +"."

            text_all = "## Conversation\n"+conversations[0] +"\n"+ photo_a_description +"\n"+ conversations[1] +"\n"+ photo_b_description +"\n"+ "\n".join(conversations[2:])
            text_all += f"\n## Problem\n{problem}\n## Choices\n{choices}\n## Solution\n{solution}."
            
            data_list.append(text_all)

    return data_list

# 학습 데이터 준비 (예시)
texts = load_data(DATA_PATH)

# 데이터셋 생성
TOTAL_DATA_COUNT = len(texts)
TRAIN_DATA_COUNT = int(TOTAL_DATA_COUNT * 0.8)
TEST_DATA_COUNT = TOTAL_DATA_COUNT - TRAIN_DATA_COUNT

EPOCH = 30
train_texts = texts[:TRAIN_DATA_COUNT]
test_texts = texts[TRAIN_DATA_COUNT:]
# print("=========start===========", processor.eos_token, processor.eos_token_id)
# print(train_texts[0])
# print("!!!tokenized!!!")
# print(processor(train_texts[0], padding="max_length", max_length=1024, truncation=True))
# print("!!!decoded!!!")
# print(processor.decode(processor(train_texts[0], padding="max_length", max_length=1024, truncation=True)['input_ids']))
# print("=========end===========")
# exit(0)

train_dataset = TextDataset(train_texts, processor, max_length=512)
eval_dataset = TextDataset(test_texts, processor, max_length=512)

print(train_texts[0])

class SaveFSDPCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):

        ### Version 1. Failed
        # Trainer에서 현재 학습 중인 모델과 tokenizer 가져오기
        # model = accelerator.unwrap_model(kwargs['model'])  # FSDP 모델 언랩
        # tokenizer = processor
        # config = model.config
        
        # # 체크포인트 저장 디렉터리
        # checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        # os.makedirs(checkpoint_dir, exist_ok=True)
        
        # # SafeTensor로 모델 가중치 저장
        # safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
        # safetensors.torch.save_file(model.state_dict(), safetensors_path)
        
        # # config.json 및 tokenizer.json 저장
        # config.save_pretrained(checkpoint_dir)
        # tokenizer.save_pretrained(checkpoint_dir)
        
        # print(f"Checkpoint saved to {checkpoint_dir}")


        ### Version 2. Success: But not sure about side effects from changing state_dict_type
        if accelerator.is_main_process:
            if trainer.is_fsdp_enabled:
                trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        
                trainer.save_model(f"./llama3.1-{NUM_PARAM}-sft-finetuned/checkpoint-{state.global_step}")

                trainer.accelerator.state.fsdp_plugin.set_state_dict_type("SHARDED_STATE_DICT")


# 학습 인자 설정
training_args = SFTConfig(
    output_dir=f"./llama3.1-{NUM_PARAM}-sft-finetuned",
    num_train_epochs=EPOCH,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=2e-5,
    fp16=True,
    gradient_checkpointing=False,
    max_seq_length=512,
    resume_from_checkpoint=True
)

# 데이터 콜레이터 설정
response_template = "## Solution"
data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=processor, mlm=False)

aim_callback = AimCallback(experiment='huggingface_experiment_sft')

accelerator = Accelerator()

process_rank = accelerator.process_index  # 현재 프로세스의 rank
local_rank = accelerator.local_process_index  # 로컬 머신에서의 rank
world_size = accelerator.num_processes  # 전체 프로세스 수

class GenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, num_samples=3, max_length=512):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = num_samples
        self.max_length = max_length
    
    def on_evaluate(self, args, state, control, model, **kwargs):
        # 평가 단계에서 샘플 생성

        model.eval()
        samples = random.sample(range(len(self.eval_dataset)), self.num_samples)
        
        print("\n=== Generation Examples ===")
        correct_count = 0
        for idx in samples:
            # 입력 텍스트 준비
            input_text, gt = self.eval_dataset[idx].split(response_template)
            input_text += response_template

            # 토큰화 및 생성
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,  # 종료 토큰 ID
                    early_stopping=True,
                    num_beams=3,
                    repetition_penalty=1.5,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\n---- Input:\n{input_text}")
            print(f"\n---- Ground Truth:\n{gt}")
            print(f"\n---- Generated:\n{generated_text[len(input_text):]}")
            print(generated_text[len(input_text):][:2].strip(), gt[:2].strip())
            if generated_text[len(input_text):][:2].strip() == gt[:2].strip():
                
                correct_count += 1
            
            print("\n" + "="*50)
        
        print(f"\n\nCorrect count: {correct_count}/{self.num_samples}")
        
        model.train()


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[aim_callback, SaveFSDPCheckpointCallback()], # 
    formatting_func=lambda x: x,
)

# 학습 시작
trainer.train()
trainer.evaluate()
print("SAVE_MODEL!")


checkpoint = None
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
trainer.train(resume_from_checkpoint=checkpoint)

# saving final model
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
trainer.save_model(f"./llama3.1-{NUM_PARAM}-sft-finetuned")



####
# if process_rank == 0:
#     model.eval()
#     samples = list(range(len(test_texts))) 
#     print("\n=== Generation Examples ===")
#     correct_count = 0
#     for idx in samples:
#         # 입력 텍스트 준비
#         input_text, gt = test_texts[idx].split(response_template)
#         input_text += response_template

#         # 토큰화 및 생성
#         inputs = processor(input_text, return_tensors="pt", truncation=True)
#         inputs = {k: v.to(model.device) for k, v in inputs.items()}
#         print(model.device)
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_length=1024,
#                 num_return_sequences=1,
#                 temperature=0.7,
#                 do_sample=True,
#                 pad_token_id=processor.pad_token_id,
#                 eos_token_id=processor.eos_token_id,  # 종료 토큰 ID
#                 early_stopping=True,
#                 num_beams=3,
#                 repetition_penalty=1.5,
#             )
        
#         generated_text = processor.decode(outputs[0], skip_special_tokens=True)
        
#         print(f"\n---- Input:\n{input_text}")
#         print(f"\n---- Ground Truth:\n{gt}")
#         print(f"\n---- Generated:\n{generated_text[len(input_text):]}")
#         print(generated_text[len(input_text):][:2].strip(), gt[:2].strip())
#         if generated_text[len(input_text):][:2].strip() == gt[:2].strip():
            
#             correct_count += 1
        
#         print("\n" + "="*50)

#     print(f"\n\nCorrect count: {correct_count}/{len(test_texts)}")