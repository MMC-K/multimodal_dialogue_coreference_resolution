from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    MllamaForConditionalGeneration, AutoProcessor
)
from datasets import load_dataset
from accelerate import Accelerator
import torch
from torch.utils.data import Dataset
import os

# 모델과 토크나이저 초기화
model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"  # 또는 다른 Llama-2 모델
# tokenizer = AutoProcessor.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

model = MlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 토크나이저 설정
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

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

# 학습 데이터 준비 (예시)
texts = [
    "여기에 실제 학습할 텍스트 데이터를 넣으세요",
    "각 텍스트는 리스트의 요소로 들어갑니다",
    # ... 더 많은 텍스트 데이터
]

# 데이터셋 생성
train_dataset = TextDataset(texts, processor)

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./llama3.2-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-5,
    fp16=True,
    gradient_checkpointing=True,
)

# 데이터 콜레이터 설정
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer 초기화 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)
exit(0)
# 학습 시작
trainer.train()

# 모델 저장
trainer.save_model("./llama3.2-finetuned-final")
