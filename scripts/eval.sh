# 1B
for MODEL_PARAMS in "1B" "3B" #"8B"
do
    CVDS=0
    for s in 0 152 326 478 630
    do
        
        if [ "$MODEL_PARAMS" != "8B" ]; then
            MODEL_PATH="./llama3.2-${MODEL_PARAMS}-sft-finetuned/checkpoint-$s"
        else
            MODEL_PATH="./llama3.1-${MODEL_PARAMS}-sft-finetuned/checkpoint-$s"
        fi
        
        if [ "$s" = "0" ]; then
            if [ "$MODEL_PARAMS" != "8B" ]; then
                MODEL_PATH="meta-llama/Llama-3.2-${MODEL_PARAMS}"
            elif [ "$MODEL_PARAMS" = "8B" ]; then
                MODEL_PATH="meta-llama/Llama-3.1-8B"
            fi
        fi

        echo ${CVDS}, ${MODEL_PATH}
        CUDA_VISIBLE_DEVICES="$CVDS" python -u llama_eval.py $MODEL_PATH | tee logs/"eval_step_${s}_${MODEL_PARAMS}.txt" &
        CVDS=$((CVDS+1))
    done
    wait
done
