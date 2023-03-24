#!/bin/bash
HEAD_NODE_IP=$1
TARGET_FOLDER=$2
WORKERS_PER_NODE=$3
MODEL_SIZE=$4
ENV_DIR=$5

source "${ENV_DIR}/bin/activate"

printenv
nvidia-smi

read -r -d '' cmd << EOF
python -m torch.distributed.run \
--nnodes 2 \
--nproc_per_node ${WORKERS_PER_NODE} \
--rdzv_id 6969 \
--rdzv_backend c10d \
--rdzv_endpoint ${HEAD_NODE_IP}:29500 \
/ssd005/projects/llm/llama/llama/example.py \
--max_batch_size 8 \
--max_seq_len 256 \
--ckpt_dir ${TARGET_FOLDER}/${MODEL_SIZE} \
--tokenizer_path ${TARGET_FOLDER}/tokenizer.model
EOF

echo "${cmd}"

bash -c "${cmd}"
