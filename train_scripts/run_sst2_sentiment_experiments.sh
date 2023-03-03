#!/bin/bash

mkdir -p /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/sst2/

rates=(0.3 0.1 0.001 0.01 0.0005 0.005)
exps=(all_finetune classifier_finetune input_finetune output_finetune)

for i in ${!rates[@]};
do
	rate=${rates[$i]}
    for j in ${!exps[@]};
    do
        exp=${exps[$j]}
        mkdir -p /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/sst2/${exp}
        mkdir -p /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/sst2/${exp}/${rate}
        sbatch src/reference_implementations/run_singlenode_prompt.slrm \
            src/reference_implementations/prompt_zoo/finetuning_sentiment.sh \
            ./torch-prompt-tuning-exps-logs \
            ${exp} \
            sst2
            /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/sst2/${exp}/${rate}
            ${rate}
    done
done


rates=(0.5 0.3 0.1)
exps=(soft_prompt_finetune)

for i in ${!rates[@]};
do
	rate=${rates[$i]}
    for j in ${!exps[@]};
    do
        exp=${exps[$j]}
        mkdir -p /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/sst2/${exp}
        mkdir -p /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/sst2/${exp}/${rate}
        sbatch src/reference_implementations/run_singlenode_prompt.slrm \
            src/reference_implementations/prompt_zoo/soft_prompt_sentiment.sh \
            ./torch-prompt-tuning-exps-logs \
            ${exp} \
            sst2 \
            /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/sst2/${exp}/${rate} \
            ${rate} \
            100
    done
done
