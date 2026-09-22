#!/usr/bin/env bash
# End-to-end SWE coding-agent RL on 8 nodes (64 train + 32 rollout = 96 GPUs total)
# with DeepSeek-V3-0321 32B-A4B, FULLY-ASYNC non-colocated variant.
#
# Differences from run_021_32b_a4b_scaleswe_openhands_8nodes.sh (synchronous):
#   - Driver: train_async.py (pipelined async; colocation disallowed)
#   - Non-colocated: actor GPUs and rollout GPUs are disjoint sets
#   - --rollout-function-path: fully_async_rollout (persistent background worker)
#   - Mismatch metrics enabled (--get-mismatch-metrics, no --use-tis)
#   - No eval args (fully-async entrypoint rejects evaluation mode)
#
# Ray cluster is assumed already up. Point RAY_API_SERVER_ADDRESS at the head
# dashboard (default http://127.0.0.1:8265).

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# Source model architecture (40 layers, MLA, 80 experts, topk 7, softmax routing).
source "${SLIME_DIR}/scripts/models/021-32B-A4B.sh"

# Params
export ROLLOUT_BATCH_SIZE=16
export ROLLOUT_GROUP_SIZE=8
export GLOBAL_BATCH_SIZE=128
export ROLLOUT_TP_SIZE=2
export ROLLOUT_DP_SIZE=8
export ROLLOUT_EP_SIZE=8
export ROLLOUT_MEM_UTILIZATION=0.8
export NUM_EPOCH=1
export SGLANG_SERVER_CONCURRENCY=4

# ============ model parallelism (training actors) ============
# 64 actor GPUs: 8 nodes x 8 GPUs. TP=2, CP=8, EP=8 unchanged from sync run.
export TP_SIZE="${TP_SIZE:-2}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-8}"
export ETP_SIZE="${ETP_SIZE:-1}"

# ============ rollout engine (32 dedicated GPUs, non-colocated) ============
# 32 rollout GPUs / TP=2 per engine = 16 engines.
# Concurrency = SGLANG_SERVER_CONCURRENCY * num_engines = 4 * 16 = 64 in-flight trajectories.
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-32}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"
ROLLOUT_DP_SIZE="${ROLLOUT_DP_SIZE:-8}"
ROLLOUT_EP_SIZE="${ROLLOUT_EP_SIZE:-8}"
ROLLOUT_MEM_UTILIZATION="${ROLLOUT_MEM_UTILIZATION:-0.8}"

# ============ startup validation ============
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${MLP_WORKER_NUM:-8}}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
ACTOR_NUM_GPUS=$(( ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE ))

if (( ROLLOUT_NUM_GPUS % ROLLOUT_TP_SIZE != 0 )); then
    echo "ERROR: ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS} is not divisible by ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE}" >&2
    exit 1
fi

TOTAL_GPUS=$(( ACTOR_NUM_GPUS + ROLLOUT_NUM_GPUS ))
echo "GPU layout: ${ACTOR_NUM_GPUS} actor + ${ROLLOUT_NUM_GPUS} rollout = ${TOTAL_GPUS} total"
echo "Rollout engines: $(( ROLLOUT_NUM_GPUS / ROLLOUT_TP_SIZE )), concurrency: $(( SGLANG_SERVER_CONCURRENCY * ROLLOUT_NUM_GPUS / ROLLOUT_TP_SIZE ))"

# ============ context length ============
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-131072}"
MAX_GEN_LEN="${MAX_GEN_LEN:-32768}"

# ============ paths — override before launching ============
HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/DeepSeek-V3-0321}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/path/to/DeepSeek-V3-0321_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/path/to/swe_train.jsonl}"

EXP_TAG="${EXP_TAG:-agent_only_async}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${SLIME_DIR}/runs/${EXP_TAG}_${STAMP}}"

# ============ logging ============
LOG_DIR="${RUN_ROOT}"
mkdir -p "${LOG_DIR}/rollout_dumps"
LOG_FILE="${LOG_DIR}/run.log"
echo "======================================================================"
echo "Training log: ${LOG_FILE}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "======================================================================"
export TENSORBOARD_DIR=${LOG_DIR}/tensorboard

# MODEL_ARGS already populated by the sourced model script above.

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_MODEL_PATH}"
   --load "${RUN_ROOT}/checkpoints"
   --save "${RUN_ROOT}/checkpoints"
   --save-interval 10
)

ROLLOUT_ARGS=(
   --custom-generate-function-path examples.coding_agent_rl.generate.generate
   --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   # --num-rollout 100
   --num-epoch ${NUM_EPOCH}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${ROLLOUT_GROUP_SIZE}
   --rollout-max-context-len ${MAX_CONTEXT_LEN}
   --rollout-max-response-len ${MAX_GEN_LEN}
   --rollout-temperature 1.0
   --rollout-stop-token-ids 128012
   --num-steps-per-rollout 1
   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --micro-batch-size 1
   --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
   --rollout-shuffle
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --sequence-parallel
   --pipeline-model-parallel-size ${PP_SIZE}
   --context-parallel-size ${CP_SIZE}
   --expert-model-parallel-size ${EP_SIZE}
   --expert-tensor-parallel-size ${ETP_SIZE}
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --max-tokens-per-gpu $((MAX_CONTEXT_LEN / CP_SIZE))
   --log-probs-chunk-size 1024
   --use-dynamic-batch-size
)

ALGO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
   --rollout-num-gpus-per-engine ${ROLLOUT_TP_SIZE}
   --sglang-mem-fraction-static ${ROLLOUT_MEM_UTILIZATION}
   --sglang-enable-dp-attention
   --sglang-dp-size ${ROLLOUT_DP_SIZE}
   --sglang-ep-size ${ROLLOUT_EP_SIZE}
   --sglang-enable-dp-lm-head
   --sglang-moe-dense-tp-size 1
   --sglang-tool-call-parser glm47
   --sglang-reasoning-parser deepseek-r1
   --sglang-enable-mixed-chunk
   --sglang-enable-cache-report
   --router-policy consistent_hashing
   --sglang-server-concurrency ${SGLANG_SERVER_CONCURRENCY}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # MLA + flash attention: leave disabled until verified for this model family.
   # --attention-backend flash
   --moe-token-dispatcher-type flex
   --moe-enable-deepep
   # Note: no --colocate — train_async.py requires non-colocated placement.
   --use-tensorboard
)

# ============ mismatch measurement (metrics only, no loss change) ============
# --get-mismatch-metrics computes KL, perplexity diff, log-prob abs diff, and
# mis_* distribution metrics without altering the training objective.
# --use-tis is intentionally omitted; add it (and tune mis.yaml bounds) once
# you have observed the mismatch distribution from this run.
# CP=8 is active so the CP-aware wrapper compute_mis_weights_with_cp is required.
MISMATCH_ARGS=(
   --get-mismatch-metrics
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

# ============ ray cluster network ============
export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(hostname -I | awk '{print $1}')}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-6379}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"

# ============ SWE / OpenHands rollout knobs ============
export SWE_AGENT="${SWE_AGENT:-openhands}"
export SWE_TRAIN_PROTOCOL="${SWE_TRAIN_PROTOCOL:-scaleswe}"
export E2B_API_KEY="${E2B_API_KEY:-e2b_0000000000000000000000000000000000000000}"
export E2B_DOMAIN="${E2B_DOMAIN:?set E2B_DOMAIN to your E2B/Kruise gateway host}"
export SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY="${SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY:-image}"
export SLIME_SANDBOX_TEMPLATE="${SLIME_SANDBOX_TEMPLATE:-swe-openhands}"

# OpenHands agent knobs.
export SWE_OH_FAKE_USER="${SWE_OH_FAKE_USER:-0}"
export SWE_OH_MAX_ITERATIONS="${SWE_OH_MAX_ITERATIONS:-100}"
export SWE_OH_TOOLS="${SWE_OH_TOOLS:-file_editor,terminal,task_tracker,think,finish}"
export SWE_PROMPT_TEMPLATE_PATH="${SCRIPT_DIR}/prompt-template.j2"
export SWE_TRAJECTORY_DIR="${RUN_ROOT}/trajectories"
export SLIME_AGENT_OH_EXTRA_ENVS='{"OH_SEND_REASONING_CONTENT":"yes"}'

# ADAPTER_PUBLIC_HOST must be routable from inside the sandbox (not 127.0.0.1).
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}}"
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18001}"

export SWE_AGENT_TIME_BUDGET_SEC="${SWE_AGENT_TIME_BUDGET_SEC:-3600}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-600}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-16}"

# ============ proxy bypass for in-cluster traffic ============
export no_proxy="127.0.0.1,${MASTER_ADDR},${ADAPTER_PUBLIC_HOST}"
export NO_PROXY="${no_proxy}"

cd "${SLIME_DIR}"

# ============ runtime env propagated to ray workers ============
export SLIME_DIR
RUNTIME_ENV_JSON=$(python3 - <<PY
import json, os
keys = (
    "no_proxy", "NO_PROXY",
    "E2B_DOMAIN", "E2B_API_KEY", "ADAPTER_PUBLIC_HOST",
    "ADAPTER_BIND_HOST", "ADAPTER_PORT",
    "TENSORBOARD_DIR",
)
env = {k: os.environ[k] for k in keys if k in os.environ}
for k, v in os.environ.items():
    if k.startswith("SLIME_") or k.startswith("SWE_"):
        env[k] = v
env["MASTER_ADDR"] = os.environ["MASTER_ADDR"]
env["MASTER_PORT"] = os.environ.get("MASTER_PORT", "")
env["GLOO_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["TP_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["NCCL_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"]
env["PYTHONPATH"] = f"/root/Megatron-LM/:{os.environ['SLIME_DIR']}"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
env["NCCL_NVLS_ENABLE"] = "0"
print(json.dumps({"env_vars": env}))
PY
)

ray job submit --address="${RAY_API_SERVER_ADDRESS:-http://127.0.0.1:8265}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -u "${SLIME_DIR}/train_async.py" \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${ALGO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${MISMATCH_ARGS[@]}" \
   2>&1 | tee "${LOG_FILE}"

echo "RUN_ROOT=${RUN_ROOT}"
