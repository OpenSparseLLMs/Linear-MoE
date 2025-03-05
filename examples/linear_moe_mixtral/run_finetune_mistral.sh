#!/bin/bash
set -e

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
LINEAR_MOE_PATH=$( dirname $( dirname ${CURRENT_DIR}))
MEGATRON_PATH=${LINEAR_MOE_PATH}/third_party/Megatron-LM-0.9.0
FLA_PATH=${LINEAR_MOE_PATH}/third_party/flash-linear-attention-1018
echo $MEGATRON_PATH
echo $FLA_PATH
export PYTHONPATH=${MEGATRON_PATH}:${LINEAR_MOE_PATH}:$PYTHONPATH
export PYTHONPATH=${FLA_PATH}:${LINEAR_MOE_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_ENDPOINT=https://hf-mirror.com

ENV=dsw
MODEL_SIZE=Small
BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1
LR=1e-5
MIN_LR=1e-6
SEQ_LEN=128
PAD_LEN=128
EXTRA_VOCAB_SIZE=0
PR=bf16
TP=1
PP=1
CP=1
EP=1
AC=sel
DO=true
FL=false
FU=false
SP=false
TE=false
MB=false
MOE=true
USE_GEMM=false
TOKEN_DROPPING=false
TRAIN_CAPACITY_FACTOR=1.25
EVAL_CAPACITY_FACTOR=2.0
SAVE_INTERVAL=100000
DATASET_PATH=datasets/mistral-datasets/wudao_mistralbpe_content_document
PRETRAIN_CHECKPOINT_PATH=models/Mistral-7B-v0.1
TRAIN_TOKENS=100000000
WARMUP_TOKENS=10000
OUTPUT_BASEPATH=./output

LA_MODULE="gla"
BASE_MODEL="mixtral"

# for models except mamba2
LAYER_TYPE_LIST="LLLLLLLL"
# LAYER_TYPE_LIST="LLLLLLLLLLLLLLLL"
# LAYER_TYPE_LIST="LLLNLLLNLLLN"
# LAYER_TYPE_LIST="LLLNLLLNLLLNLLLN"

# for only mamba2, MLP layers are fixed behind mamba or attention layers. M: mamba layer, *: attention layer
# for pure_mamba2
HYBRID_OVERRIDE_PATTERN="MMMMMMMM"
# HYBRID_OVERRIDE_PATTERN="MMMMMMMMMMMMMMMM"
# for hybrid_mamba2
# HYBRID_OVERRIDE_PATTERN="MMM*MMM*MMM*"
# HYBRID_OVERRIDE_PATTERN="MMM*MMM*MMM*MMM*"

# # Turn on --megatron-hybrid-mamba-method to use the logic in Megatron-LM.
# HYBRID_OVERRIDE_PATTERN="M-M-M-*-M-M-M-*-M-M-M-*-"
# HYBRID_OVERRIDE_PATTERN="M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-"

# # SSM
# linear_moe_options=" \protobuf                      3.18.3
#         --use-la-module \
#         --use-cache \
#         --la-module pure_mamba2 \
#         --base-model mixtral \
#         "

# Linear Attention
linear_moe_options=" \
        --use-la-module \
        --la-module ${LA_MODULE} \
        --la-mode fused_chunk \
        --base-model ${BASE_MODEL} \
        --la-feature-map swish \
        --la-output-norm rmsnorm \
        --la-gate-fn swish \
        --layer-type-list ${LAYER_TYPE_LIST} \
        "

# # Linear RNN
# linear_moe_options=" \
#         --use-la-module \
#         --use-cache \
#         --la-module rwkv6 \
#         --la-mode chunk \
#         --base-model mixtral \
#         --la-output-norm groupnorm \
#         --la-gate-fn swish \
#         "

if [ $MB = true ]; then
    linear_moe_options="${linear_moe_options} \
        --moe-megablocks \
        "
fi

if [ $TOKEN_DROPPING = true ]; then
    linear_moe_options="${linear_moe_options} \
        --moe-train-capacity-factor ${TRAIN_CAPACITY_FACTOR} \
        --moe-eval-capacity-factor ${EVAL_CAPACITY_FACTOR} \
        --moe-token-dropping \
        "
fi

if [ $USE_GEMM = true ]; then
    linear_moe_options="${linear_moe_options} \
        --moe-grouped-gemm \
        "
fi

if [ $ENV = dsw ]; then
export CUDA_VISIBLE_DEVICES=1
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1
TOTAL_GPUS=$(($GPUS_PER_NODE*$NNODES))

elif [ $ENV = dlc ]; then

NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
TOTAL_GPUS=$(($GPUS_PER_NODE*$NNODES))

fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

if [ $MODEL_SIZE = Small ]; then

NUM_LAYERS=8
HIDDEN_SIZE=1024
NUM_ATTN_HEADS=8
INTERMEDIATE_SIZE=14336
MAX_POSITION_EMBEDDINGS=32768
SLW=4096

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups 8"

fi

if [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=14336
MAX_POSITION_EMBEDDINGS=32768
SLW=4096

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups 8"

fi

if [ $AC = full ]; then
    activation_checkpoint_options=" \
        --recompute-num-layers 1 \
		    --recompute-method uniform \
		    --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
                    "
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16 \
            --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16
        --fp8-hybrid \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024 \
        --transformer-impl transformer_engine"
fi

if [ $DO = true ]; then
    do_options=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

if [ $FL = true ]; then
    flash_options=" \
		    --use-flash-attn"

elif [ $FL = false ]; then
    flash_options=" \
                    "
fi

if [ $TE = true ]; then
    te_options=" \
		    --transformer-impl transformer_engine"

elif [ $TE = false ]; then
    te_options=" \
                    "
fi

if [ $MOE = true ]; then
    moe_options=" \
		    --moe-router-topk 2 \
		    --num-experts 8 \
		    --moe-aux-loss-coeff 1e-2 \
		    --expert-model-parallel-size 1 \
		    --moe-router-load-balancing-type aux_loss"

elif [ $MOE = false ]; then
    moe_options=" \
                    "
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_options=" \
            --load $PRETRAIN_CHECKPOINT_PATH"
fi

EP=$(($TOTAL_GPUS/$TP/$PP))

TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

NAME="${ENV}-pretrain-megatron-gpt3-${MODEL_SIZE}-lr-${LR}-bs-${BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}-tp-${TP}-pp-${PP}-ac-${AC}-do-${DO}-sp-${SP}-tt-${TRAIN_TOKENS}-wt-${WARMUP_TOKENS}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}

LOG_FILE="${OUTPUT_BASEPATH}/log/${current_time}_${NAME}.log"

SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

megatron_options="  \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --data-path ${DATASET_PATH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --init-method-std 0.006 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --split 99,1,0 \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --max-padding-length ${PAD_LEN} \
        --sliding-window ${SLW} \
        --log-interval 1 \
        --eval-interval 10000 \
        --eval-iters 10 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --no-load-optim \
        --no-load-rng \
        --num-workers 0 \
        --seed 1234 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type MistralTokenizer \
        --dataset LLama-Pretrain-Idxmap \
        --swiglu \
        --use-rotary-position-embeddings \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --disable-bias-linear-fc \
        --disable-bias-attn-fc \
        --normalization LayerNorm \
        --no-masked-softmax-fusion \
        --no-position-embedding \
        --use-mcore-models \
        --no-rope-fusion \
        --distributed-timeout-minutes 6000 \
        --transformer-impl transformer_engine \
        "



run_cmd="torchrun $DISTRIBUTED_ARGS pretrain_mcore_mistral.py
 ${megatron_options} ${pr_options} ${load_options} ${te_options} ${activation_checkpoint_options} ${do_options} ${flash_options} ${sp_options} ${gqa_options} ${moe_options} ${linear_moe_options} 2>&1 | sudo tee -a $LOG_FILE"

echo ${run_cmd}
eval ${run_cmd}
set +x

# note

# 在使用layer_specs.py中FusedLayerNorm时，这里的--normalization必须设为LayerNorm，否则报错
# --normalization RMSNorm \
# --normalization LayerNorm \