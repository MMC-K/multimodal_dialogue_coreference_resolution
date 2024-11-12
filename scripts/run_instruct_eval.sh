CVDS=0
for s in 0 152 326 478 630
do
    MODEL_PATH="./llama3.1-8B-instruct-sft-finetuned-chat-template-True/checkpoint-$s"
    
    if [ "$s" = "0" ]; then
        MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
    fi

    echo ${CVDS}, ${MODEL_PATH}
    CUDA_VISIBLE_DEVICES="$CVDS" python -u llama_instruct_eval.py $MODEL_PATH | tee logs/"eval_step_${s}_8B_instruct.txt" &
    CVDS=$((CVDS+1))
done


wait