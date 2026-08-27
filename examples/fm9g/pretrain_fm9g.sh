#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=$1 #<Specify path>
TENSORBOARD_LOGS_PATH="./tensorboard" #<Specify path>
VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
# DATA_PATH="data/wikitext-2/wikitext2_text_document_text_document" #<Specify path and file prefix>_text_document
# DATA_PATH="data/wikitext-103/processed/wikitext-103_text_document"
DATA_PATH="data/openwebtext/processed/openwebtext_text_document"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    # --num-layers 16 
    # --hidden-size 3584 
    # --num-attention-heads 28 
    # --max-position-embeddings 32768 
    # --ffn-hidden-size 18944 
    # --seq-length 4096 
    # --vocab-size 73448 
    # --num-layers 16 
    # --hidden-size 3584 
    # --ffn-hidden-size 18944 
    # --num-attention-heads 28 
    # --num-query-groups 4 
    # --vocab-size 73448 
    # --max-position-embeddings 32768 
    # --seq-length 4096 

    # AI-config ~ 1.6B
    # --num-layers 12 
    # --hidden-size 3072 
    # --ffn-hidden-size 12288 
    # --num-attention-heads 24 
    # --num-query-groups 4 
    # --vocab-size 73448 
    # --max-position-embeddings 8192 
    # --seq-length 2048 
    
    # 7B config
    # --num-layers 28 
    # --hidden-size 3584 
    # --ffn-hidden-size 18944 
    # --num-attention-heads 28 
    # --num-query-groups 4 
    # --vocab-size 73448 
    # --max-position-embeddings 32768 
    # --seq-length 32768 
    
    # 4B config
    # --num-layers 62 
    # --hidden-size 2560 
    # --ffn-hidden-size 6400 
    # --num-attention-heads 40 
    # --num-query-groups 4 
    # --vocab-size 73448 
    # --max-position-embeddings 32768 
    # --seq-length 32768 

    # 2.4B config
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
    # --global-batch-size 1536 
    # --train-iters 500000 
    # --weight-decay 0.1 
    # --adam-beta1 0.9 
    # --adam-beta2 0.95 
    # --init-method-std 0.006 
    # --clip-grad 1.0 
    # --fp16
    # --lr 6.0e-5 
    # --lr-decay-style cosine 
    # --min-lr 6.0e-6
    # --lr-warmup-fraction .001 
    # --lr-decay-iters 430000 

    # --precision bf16 
    --distributed-backend nccl 
    # --num-layers-per-virtual-pipeline-stage 4 
    --data-path data/wikitext-2/wikitext2_text_document_text_document 
    --tokenizer-type SentencePieceTokenizer 
    --tokenizer-model jiuge/tokenizer/tokenizer.model 
    --lr 1e-4 
    --lr-decay-style cosine 
    --lr-warmup-fraction 0.02 
    --bf16
    --train-iters 1000
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --adam-eps 1e-6 
    --use-distributed-optimizer
    --log-interval 1 
    --log-throughput
    # --no-overlap-p2p-communication 
    # --recompute-granularity selective 
    --use-flash-attn 
    --pos-bias-type rotary
    
    --overlap-grad-reduce 
    --overlap-param-gather 
    
    # --use-arc-topk
    # --arc-topk-compression-ratio 0.1
    # --use-error-feedback
    --optimizer-cpu-offload # offloading
    --optimizer-offload-fraction 1
    --use-precision-aware-optimizer

    # --use-grad-quantization
    # --grad-quantization-dtype int8
    # --use-bitscom
    
    # --adjust-compression-ratio
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1 
	--pipeline-model-parallel-size 1 
    

    # --use-torch-fsdp2 
    # --ckpt-format torch_dist 
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    # --vocab-file $VOCAB_FILE 
    # --merge-file $MERGE_FILE 
    --split 949,50,1
)

PROFILING_ARGS=(
    --profile
    --profile-step-start 8
    --profile-step-end 10
    --profile-ranks 0
    --use-pytorch-profiler
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000 
    --eval-interval 100 
    # --save $CHECKPOINT_PATH 
    # --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

echo "the final cmd is:"
echo "------------------------------------------------------------"
echo "torchrun ${DISTRIBUTED_ARGS[@]} pretrain_fm9g.py ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${DATA_ARGS[@]} \
"
echo "------------------------------------------------------------"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_fm9g.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${PROFILING_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${DATA_ARGS[@]} \

# python pretrain_fm9g.py \
#     ${GPT_MODEL_ARGS[@]} \
#     ${TRAINING_ARGS[@]} \
#     ${MODEL_PARALLEL_ARGS[@]} \
#     ${DISTRIBUTED_ARGS[@]} \
