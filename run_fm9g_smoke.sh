#!/bin/bash
# fm9g 冒烟测试训练脚本
#  - 300 iters, 不保存 checkpoint
#  - 每 10 个 iter 打印一次 loss
#  - 训练结束后把 tensorboard 文件归档到 ./log/<YYYYmmdd_HHMMSS>/

# 切到脚本所在目录（repo 根目录）
cd "$(dirname "$0")"

# 激活 bitscom conda 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bitscom

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

TENSORBOARD_DIR="./tensorboard"
LOGS_ROOT="./log"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# 2.4B config（当前已验证可跑起来的最小配置）
GPT_MODEL_ARGS=(
    --num-layers 20
    --hidden-size 2304
    --ffn-hidden-size 5760
    --num-attention-heads 36
    --num-query-groups 4
    --vocab-size 122753
    --max-position-embeddings 2048
    --seq-length 2048

    --untie-embeddings-and-output-weights
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --distributed-backend nccl
    --tokenizer-type SentencePieceTokenizer
    --tokenizer-model jiuge/tokenizer/tokenizer.model
    --lr 1e-4
    --lr-decay-style cosine
    --lr-warmup-fraction 0.02
    --bf16
    --train-iters 50
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-6
    --use-distributed-optimizer
    --log-throughput
    --use-flash-attn
    --pos-bias-type rotary

    --overlap-grad-reduce
    --overlap-param-gather

    --optimizer-cpu-offload # offloading
    --optimizer-offload-fraction 1
    --use-precision-aware-optimizer

    # ---- bitscom: 低比特量化 All-Reduce / Reduce-Scatter（取消注释 --use-bitscom 即启用）----
    --use-bitscom                          # 启用 bitscom 后端（ProcessGroupLowBit）
    # --bitscom-bitwidth 4                   # 量化位宽 1/2/4/8/12/16；<8 走低比特量化 all2all+allgather 路径，>=8 回退标准 NCCL
    # --bitscom-error-feedback               # 开启 Stage-1 error feedback（默认关闭）
    # --bitscom-error-feedback-mode legacy   # EF 模式：none / legacy / ef21 / ef21_plus
    # --bitscom-block-size 256               # 分块量化块大小
    # ---- bitscom 稀疏 ARC-Top-K（可选，默认关闭）----
    --bitscom-sparse-enabled
    --bitscom-sparse-projection-rank 4     # 随机投影秩
    --bitscom-sparse-compression-ratio 0.1 # 优先行比例
    --bitscom-sparse-priority-mode 0       # 0=kFull 全精度 / 1=kQuantize 量化 / 2=kDiscard 丢弃
    --bitscom-sparse-priority-quantize-bitwidth 4
    --bitscom-sparse-non-priority-mode 2   # 0=kFull / 1=kQuantize / 2=kDiscard
    --bitscom-sparse-non-priority-quantize-bitwidth 4
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path data/openwebtext/processed/openwebtext_text_document
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --eval-interval 100
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_DIR
)

PROFILING_ARGS=(
    --profile
    --profile-step-start 8
    --profile-step-end 10
    --profile-ranks 0
    --use-pytorch-profiler
)

echo "the final cmd is:"
echo "------------------------------------------------------------"
echo "torchrun ${DISTRIBUTED_ARGS[@]} pretrain_fm9g.py ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${PROFILING_ARGS[@]} \
    ${DATA_ARGS[@]} \
"
echo "------------------------------------------------------------"

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_fm9g.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${PROFILING_ARGS[@]} \
    ${DATA_ARGS[@]} \

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    # 训练正常结束：归档 tensorboard 到 ./log/<时间戳>/
    if [ -d "$TENSORBOARD_DIR" ]; then
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        LOG_DIR="$LOGS_ROOT/$TIMESTAMP"
        mkdir -p "$LOG_DIR"
        mv "$TENSORBOARD_DIR" "$LOG_DIR/"
        echo ""
        echo "tensorboard 文件已归档到: $LOG_DIR/tensorboard"
        echo "查看方式: tensorboard --logdir $LOG_DIR/tensorboard"
    else
        echo ""
        echo "未发现 tensorboard 目录 $TENSORBOARD_DIR（训练可能未开始或未启用 tensorboard 日志）"
    fi
else
    # 训练失败：tensorboard 留在原地方便排查，不归档
    echo ""
    echo "训练失败（exit code=$TRAIN_EXIT_CODE），tensorboard 不归档，留在 $TENSORBOARD_DIR 以便排查"
fi

exit $TRAIN_EXIT_CODE
