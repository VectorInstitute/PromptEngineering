#!/bin/bash
WORKERS_PER_NODE=$1
ENV_DIR=$2
NNODES=$3
MAX_BATCH_SIZE=$4
MAX_SEQ_LEN=$5
PYTHON_SCRIPT=$6
CKPT_DIR=$7
TOKENIZER_PATH=$8
RDVZ_ID=$9
RDVZ_BACKEND=${10}
RDVZ_ENDPOINT=${11}

source "${ENV_DIR}/bin/activate"

printenv
nvidia-smi

read -r -d '' cmd << EOF
python -m torch.distributed.run \
--nnodes ${NNODES} \
--nproc_per_node ${WORKERS_PER_NODE} \
--rdzv_id ${RDVZ_ID} \
--rdzv_backend ${RDVZ_BACKEND} \
--rdzv_endpoint ${RDVZ_ENDPOINT} \
${PYTHON_SCRIPT} \
--max_batch_size ${MAX_BATCH_SIZE} \
--max_seq_len ${MAX_SEQ_LEN} \
--ckpt_dir ${CKPT_DIR} \
--tokenizer_path ${TOKENIZER_PATH}
EOF

echo "${cmd}"

bash -c "${cmd}"
