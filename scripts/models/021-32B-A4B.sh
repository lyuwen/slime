NLAYERS=40
FIRST_K_DENSE_REPLACE=1
MOE_FFN_HIDDEN=1536
MOE_SHARED_EXPERTS=1
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE=$(($MOE_FFN_HIDDEN * $MOE_SHARED_EXPERTS))

arr=()
for ((i=0; i<NLAYERS; i++)); do
  if (( i < FIRST_K_DENSE_REPLACE )); then
    arr+=(0)
  else
    arr+=(1)
  fi
done

printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

# DeepSeek-V3-0321 32B-A4B
# Architecture sourced from run_8xH20.sh (32B section).
# Uses softmax + pre-softmax routing (DeepSeek-V3 family style, distinct from
# moonlight which uses sigmoid). Rope settings follow deepseek-v3 defaults;
# adjust rotary-base / rotary-scaling-factor if your checkpoint differs.
MODEL_ARGS=(
    --disable-bias-linear
    --num-layers $NLAYERS
    --hidden-size 2048
    --ffn-hidden-size 12288
    --num-attention-heads 32
    --kv-channels 128
    --normalization RMSNorm
    --position-embedding-type rope
    --norm-epsilon 1e-6
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 128256

    --multi-latent-attention
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --qk-layernorm
    --rotary-base 5000000
    --mscale 1.0
    --mscale-all-dim 1.0
    --attention-softmax-in-fp32
    --no-rope-fusion

    # moe
    --num-experts 80
    --moe-layer-freq "$MOE_LAYER_FREQ"
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN
    --moe-router-topk 7
    --moe-shared-expert-intermediate-size $MOE_SHARED_EXPERT_INTERMEDIATE_SIZE
    --moe-router-pre-softmax
    --moe-router-score-function softmax
    --moe-router-load-balancing-type aux_loss
    --moe-token-dispatcher-type alltoall
    --moe-aux-loss-coeff 0
    --moe-router-bias-update-rate 0
    --moe-router-group-topk 1
    --moe-router-num-groups 1
    --moe-grouped-gemm
    --moe-router-dtype fp32
    --moe-permute-fusion
)
