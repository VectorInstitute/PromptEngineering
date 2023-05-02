#!/bin/bash

# This is the main directory where the model checkpoints will be saved.
# if you are working under a different directory on the vector's cluster,
# you may want to change the following sst2_path. The following directory
# is created under your username.
sst2_path="/scratch/ssd004/scratch/${USER}/sst2"
mkdir -p ${sst2_path}

rates=(0.3 0.1 0.001 0.01 0.0005 0.005)
exps=(all_finetune classifier_finetune input_finetune output_finetune)

for i in ${!rates[@]};
do
	rate=${rates[$i]}
    for j in ${!exps[@]};
    do
        exp=${exps[$j]}
        mkdir -p ${sst2_path}/${exp}
        mkdir -p ${sst2_path}/${exp}/${rate}
        sbatch src/reference_implementations/run_singlenode_prompt.slrm \
            src/reference_implementations/prompt_zoo/training_scripts/finetuning_sentiment.sh \
            ./torch-prompt-tuning-exps-logs \
            ${exp} \
            sst2 \
            ${sst2_path}/${exp}/${rate} \
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
        mkdir -p ${sst2_path}/${exp}
        mkdir -p ${sst2_path}/${exp}/${rate}
        sbatch src/reference_implementations/run_singlenode_prompt.slrm \
            src/reference_implementations/prompt_zoo/training_scripts/soft_prompt_sentiment.sh \
            ./torch-prompt-tuning-exps-logs \
            ${exp} \
            sst2 \
            ${sst2_path}/${exp}/${rate} \
            ${rate} \
            50
    done
done
