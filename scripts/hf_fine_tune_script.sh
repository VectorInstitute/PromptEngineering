#!/bin/bash
# lr_list=(0.0001 0.00001 0.000001)
lr_list=(0.00001)
# wd_list=(0.001 0.0001 0.00001)
wd_list=(0.00001)
# hf_model_name="roberta-base"
# hf_model_name="roberta-large"
# hf_model_name="facebook/opt-125m"
# hf_model_name="facebook/opt-350m"
for lr in ${lr_list[@]}
do
    for wd in ${wd_list[@]}
    do
        for es in 7
        do 
            for idx in {0..0} 
            do
                echo $lr $wd $es $idx $hf_model_name
                python3 src/reference_implementations/hugging_face_basics/hf_fine_tuning_examples/hf_fine_tune_script.py $lr $wd $es $idx $hf_model_name
            done
        done
    done
done
