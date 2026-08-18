#!/usr/bin/env bash
# End-to-end SWE coding-agent RL on 8 nodes with DeepSeek-V3-0321 32B-A4B,
# EXTERNAL-CLUSTER variant. Adapted from
# run_qwen36_35b_a3b_scaleswe_openhands_8nodes-extcluster.sh.
#
# Does NOT start/stop Ray or SSH into workers: the Ray cluster is assumed to be
# already up (head + workers joined), and we only submit the job to it. Point
# RAY_API_SERVER_ADDRESS at the running head's dashboard (default
# http://127.0.0.1:8265). See README.md for the dataset schema and env vars.

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
export ROLLOUT_DP_SIZE=1
export ROLLOUT_EP_SIZE=1
export ROLLOUT_MEM_UTILIZATION=0.8
export NUM_EPOCH=1
export SGLANG_SERVER_CONCURRENCY=4

# ============ model parallelism ============
# 8-node, 64-GPU run. PP=8 / EP=8 matches run_8xH20.sh 32B section.
# No CP: MLA + context-parallel interaction needs verification; disable for safety.
export TP_SIZE="${TP_SIZE:-2}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-8}"
export ETP_SIZE="${ETP_SIZE:-1}"

# ============ rollout engine ============
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-8}"
ROLLOUT_DP_SIZE="${ROLLOUT_DP_SIZE:-8}"
ROLLOUT_EP_SIZE="${ROLLOUT_EP_SIZE:-8}"
ROLLOUT_MEM_UTILIZATION="${ROLLOUT_MEM_UTILIZATION:-0.75}"

# ============ context length ============
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-131072}"
MAX_GEN_LEN="${MAX_GEN_LEN:-32768}"

# ============ paths — override before launching ============
HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/DeepSeek-V3-0321}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/path/to/DeepSeek-V3-0321_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/path/to/swe_train.jsonl}"

EXP_TAG="${EXP_TAG:-agent_only}"
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
   --rollout-num-gpus 64
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
   --colocate
   --use-tensorboard
)

# ============ ray cluster network ============
# Set MASTER_ADDR before the SWE block: ADAPTER_PUBLIC_HOST below falls back to it.
export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(hostname -I | awk '{print $1}')}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-6379}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"

# ============ SWE / OpenHands rollout knobs ============
export SWE_AGENT="${SWE_AGENT:-openhands}"
export SWE_TRAIN_PROTOCOL="${SWE_TRAIN_PROTOCOL:-scaleswe}"
export E2B_API_KEY="${E2B_API_KEY:-e2b_0000000000000000000000000000000000000000}"
# Host of the E2B/Kruise gateway. REQUIRED: patch_e2b (slime/agent/patch_e2b.py)
# reads os.environ['E2B_DOMAIN'] on every Ray worker at first sandbox creation
# and hard-fails with KeyError if it is unset. Must be forwarded to workers (see
# the RUNTIME_ENV_JSON keys tuple below).
export E2B_DOMAIN="${E2B_DOMAIN:?set E2B_DOMAIN to your E2B/Kruise gateway host}"
export SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY="${SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY:-image}"

# The OpenHands env is delivered as an image volume mounted at the fixed prefix
# /opt/oh-env (built by tools/oh-env-image.Dockerfile), NOT unpacked from a
# tarball at boot. SLIME_SANDBOX_TEMPLATE selects the E2B template that carries
# that mount. Forwarded to workers by the SLIME_/SWE_ prefix loop below.
export SLIME_SANDBOX_TEMPLATE="${SLIME_SANDBOX_TEMPLATE:-swe-openhands}"

# OpenHands agent knobs.
export SWE_OH_FAKE_USER="${SWE_OH_FAKE_USER:-0}"
export SWE_OH_MAX_ITERATIONS="${SWE_OH_MAX_ITERATIONS:-100}"
export SWE_OH_TOOLS="${SWE_OH_TOOLS:-file_editor,terminal,task_tracker,think,finish}"
export SWE_PROMPT_TEMPLATE_PATH="${SCRIPT_DIR}/prompt-template.j2"
export SWE_TRAJECTORY_DIR="${RUN_ROOT}/trajectories"
# Arbitrary extra env vars forwarded verbatim into the OH agent's shell (JSON obj).
# export SLIME_AGENT_OH_EXTRA_ENVS='{"HTTPS_PROXY":"http://proxy:8080"}'
export SLIME_AGENT_OH_EXTRA_ENVS='{"OH_SEND_REASONING_CONTENT":"yes"}'
# export SLIME_AGENT_OH_EXTRA_ENVS='{"OH_SEND_REASONING_CONTENT":"yes","OH_VALIDATE_TOOLCALL_PARAMS":"on"}'

# ADAPTER_PUBLIC_HOST must be routable from inside the sandbox (not 127.0.0.1).
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}}"
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18001}"

export SWE_AGENT_TIME_BUDGET_SEC="${SWE_AGENT_TIME_BUDGET_SEC:-3600}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-600}"
export SWE_EVAL_MAX_ATTEMPTS="${SWE_EVAL_MAX_ATTEMPTS:-3}"
export SWE_ROLLOUT_RETRIES="${SWE_ROLLOUT_RETRIES:-3}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-16}"

# ============ proxy bypass for in-cluster traffic ============
export no_proxy="127.0.0.1,${MASTER_ADDR},${ADAPTER_PUBLIC_HOST}"
export NO_PROXY="${no_proxy}"

cd "${SLIME_DIR}"

# ============ ray cluster (assumed already running) ============
# This variant does not start Ray or SSH into workers; it only submits to an
# existing cluster via RAY_API_SERVER_ADDRESS below.
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${MLP_WORKER_NUM:-8}}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"

# ============ runtime env propagated to ray workers ============
export SLIME_DIR
RUNTIME_ENV_JSON=$(python3 - <<PY
import json, os
# Explicit cluster/network keys that don't fit a forwarding prefix.
keys = (
    "no_proxy", "NO_PROXY",
    "E2B_DOMAIN", "E2B_API_KEY", "ADAPTER_PUBLIC_HOST",
    "ADAPTER_BIND_HOST", "ADAPTER_PORT",
    "TENSORBOARD_DIR",
)
env = {k: os.environ[k] for k in keys if k in os.environ}
# Prefix pass-through: forward every slime / SWE knob automatically so new vars
# need no per-var edit (see spec §5.1a). Matches bare SLIME_ (not just
# SLIME_AGENT_) so knobs like SLIME_SANDBOX_TEMPLATE and
# SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS are not silently dropped;
# SLIME_AGENT_OH_EXTRA_ENVS rides this rule too.
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
   -- python3 -u "${SLIME_DIR}/train.py" \
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
   2>&1 | tee "${LOG_FILE}"

echo "RUN_ROOT=${RUN_ROOT}"
