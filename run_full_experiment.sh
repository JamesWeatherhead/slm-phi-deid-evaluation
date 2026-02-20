#!/usr/bin/env bash
# =============================================================================
# SLM De-Identification Full Experiment Runner
# Target: GPU server with NVIDIA A100 GPUs
# =============================================================================
#
# This script:
#   1. Connects to the GPU server
#   2. Verifies Ollama is running
#   3. Pulls all 4 models
#   4. Runs the variance pilot
#   5. Runs the full experiment matrix
#   6. Runs evaluation and analysis
#   7. Collects results back to local machine
#
# Usage:
#   bash run_full_experiment.sh
#
# Prerequisites:
#   - sshpass installed locally (or SSH key authentication configured)
#   - Environment variables: SLM_REMOTE_HOST, SLM_REMOTE_USER
#   - Optional: SSH_CLUSTER_PASS for password-based auth
#   - Ollama running on the remote server (port 11434)
#   - Python 3.10+ with requirements.txt installed on the server
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REMOTE_HOST="${SLM_REMOTE_HOST:?Set SLM_REMOTE_HOST to your Ollama server IP}"
REMOTE_USER="${SLM_REMOTE_USER:?Set SLM_REMOTE_USER to your SSH username}"
REMOTE_PORT="11434"
OLLAMA_URL="http://${REMOTE_HOST}:${REMOTE_PORT}"

LOCAL_EVAL_DIR="${SLM_LOCAL_DIR:-$(cd "$(dirname "$0")" && pwd)}"
REMOTE_EVAL_DIR="${SLM_REMOTE_DIR:-/home/${REMOTE_USER}/slm-evaluation}"

# Models to pull (Ollama tags)
MODELS=(
    "phi4-mini:latest"
    "llama3.2:3b-instruct-q4_K_M"
    "qwen3:4b"
    "gemma3:4b"
)

MODEL_SLUGS=(
    "phi4-mini"
    "llama32-3b"
    "qwen3-4b"
    "gemma3-4b"
)

# Authenticate via SSH key (recommended) or password.
# If using password auth, set SSH_CLUSTER_PASS in your environment.
if [ -z "${SSH_CLUSTER_PASS:-}" ]; then
    # Try SSH key authentication
    SSH_CMD="ssh -o StrictHostKeyChecking=no"
    SCP_CMD="scp -o StrictHostKeyChecking=no"
else
    SSH_CMD="sshpass -p \"$SSH_CLUSTER_PASS\" ssh -o StrictHostKeyChecking=no"
    SCP_CMD="sshpass -p \"$SSH_CLUSTER_PASS\" scp -o StrictHostKeyChecking=no"
fi

# Helper function for remote commands
remote_cmd() {
    eval ${SSH_CMD} "${REMOTE_USER}@${REMOTE_HOST}" "$@"
}

remote_scp_to() {
    eval ${SCP_CMD} -r "$1" "${REMOTE_USER}@${REMOTE_HOST}:$2"
}

remote_scp_from() {
    eval ${SCP_CMD} -r "${REMOTE_USER}@${REMOTE_HOST}:$1" "$2"
}

# ---------------------------------------------------------------------------
# Step 1: Verify Ollama is running
# ---------------------------------------------------------------------------

echo "=========================================="
echo "STEP 1: Verifying Ollama on ${REMOTE_HOST}"
echo "=========================================="

if curl -s --connect-timeout 10 "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
    echo "Ollama is running at ${OLLAMA_URL}"
    OLLAMA_VERSION=$(curl -s "${OLLAMA_URL}/api/version" 2>/dev/null || echo "unknown")
    echo "Ollama version: ${OLLAMA_VERSION}"
else
    echo "ERROR: Cannot reach Ollama at ${OLLAMA_URL}"
    echo "Attempting to start Ollama on the remote server..."
    remote_cmd "tmux new-session -d -s ollama 'CUDA_VISIBLE_DEVICES=0,1,2,3,4 ollama serve' 2>/dev/null || true"
    sleep 10
    if curl -s --connect-timeout 10 "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
        echo "Ollama started successfully."
    else
        echo "ERROR: Failed to start Ollama. Please start it manually."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Step 2: Pull all models
# ---------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "STEP 2: Pulling models"
echo "=========================================="

for model in "${MODELS[@]}"; do
    echo "Pulling ${model}..."
    curl -s -X POST "${OLLAMA_URL}/api/pull" \
        -d "{\"name\": \"${model}\", \"stream\": false}" \
        --max-time 600 || {
        echo "WARNING: Pull may have timed out for ${model}. Checking if model exists..."
    }
done

# Verify all models are available
echo ""
echo "Verifying models..."
AVAILABLE_MODELS=$(curl -s "${OLLAMA_URL}/api/tags" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for m in data.get('models', []):
    print(m['name'])
" 2>/dev/null || echo "")

echo "Available models:"
echo "${AVAILABLE_MODELS}"

for model in "${MODELS[@]}"; do
    if echo "${AVAILABLE_MODELS}" | grep -q "${model}"; then
        echo "  [OK] ${model}"
    else
        echo "  [MISSING] ${model} -- attempting pull again..."
        curl -s -X POST "${OLLAMA_URL}/api/pull" \
            -d "{\"name\": \"${model}\", \"stream\": false}" \
            --max-time 600 || true
    fi
done

# ---------------------------------------------------------------------------
# Step 3: Upload evaluation code to server
# ---------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "STEP 3: Uploading evaluation code"
echo "=========================================="

remote_cmd "mkdir -p ${REMOTE_EVAL_DIR}"

# Upload src/, configs/, and data files
remote_scp_to "${LOCAL_EVAL_DIR}/src" "${REMOTE_EVAL_DIR}/"
remote_scp_to "${LOCAL_EVAL_DIR}/configs" "${REMOTE_EVAL_DIR}/"
remote_scp_to "${LOCAL_EVAL_DIR}/requirements.txt" "${REMOTE_EVAL_DIR}/"

# Upload the dataset
DATASET_DIR="${LOCAL_EVAL_DIR}/data"
remote_cmd "mkdir -p ${REMOTE_EVAL_DIR}/data"
remote_scp_to "${DATASET_DIR}/all_queries.json" "${REMOTE_EVAL_DIR}/data/"

echo "Code uploaded to ${REMOTE_EVAL_DIR}"

# Install requirements
echo "Installing Python requirements..."
remote_cmd "cd ${REMOTE_EVAL_DIR} && pip install -r requirements.txt 2>/dev/null || pip3 install -r requirements.txt"

# ---------------------------------------------------------------------------
# Step 4: Run variance pilot
# ---------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "STEP 4: Running variance pilot"
echo "=========================================="

remote_cmd "cd ${REMOTE_EVAL_DIR} && python3 -m src.pilot \
    --host http://localhost:11434 \
    --model phi4-mini \
    --prompt zero-shot-structured \
    --n-queries 100 \
    --n-runs 10 \
    --output-dir ${REMOTE_EVAL_DIR} \
    --log-file ${REMOTE_EVAL_DIR}/pilot.log"

echo "Pilot complete. Fetching results..."
remote_scp_from "${REMOTE_EVAL_DIR}/analysis/pilot_results.json" "${LOCAL_EVAL_DIR}/analysis/"

echo ""
echo "Pilot results:"
python3 -c "
import json
with open('${LOCAL_EVAL_DIR}/analysis/pilot_results.json') as f:
    d = json.load(f)
print(f\"  Byte-identical rate: {d['byte_identical_rate']*100:.1f}%\")
print(f\"  Recall mean: {d['recall_mean']:.4f} +/- {d['recall_std']:.4f}\")
print(f\"  Recommendation: {d['recommendation']}\")
print(f\"  Rationale: {d['rationale']}\")
"

# ---------------------------------------------------------------------------
# Step 5: Run full experiment (one model per GPU)
# ---------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "STEP 5: Running full experiment"
echo "=========================================="

# Prompt strategies and number of runs (must match inference_params.json runs_per_config)
PROMPTS=(zero-shot-minimal zero-shot-structured few-shot two-pass chain-of-thought)
N_RUNS=5

# Run each model on its own GPU using tmux sessions.
# Loops over all prompts x runs per model for parallel GPU execution.
for i in "${!MODEL_SLUGS[@]}"; do
    slug="${MODEL_SLUGS[$i]}"
    tag="${MODELS[$i]}"
    gpu_id=$i

    echo "Starting ${slug} on GPU ${gpu_id} (${#PROMPTS[@]} prompts x ${N_RUNS} runs)..."

    # Build the command string for all prompt x run combinations
    RUN_CMDS="export CUDA_VISIBLE_DEVICES=${gpu_id} && cd ${REMOTE_EVAL_DIR}"
    for prompt in "${PROMPTS[@]}"; do
        for run in $(seq 1 ${N_RUNS}); do
            RUN_CMDS="${RUN_CMDS} && python3 -m src.runner --model ${slug} --prompt ${prompt} --run ${run} --host http://localhost:11434 --output-dir ${REMOTE_EVAL_DIR}"
        done
    done
    RUN_CMDS="${RUN_CMDS} && echo COMPLETE: ${slug}"

    # Create a tmux session for each model
    remote_cmd "tmux new-session -d -s 'eval-${slug}' '${RUN_CMDS}' 2>&1"

    # Stagger launches by 10 seconds to avoid model loading contention
    sleep 10
done

echo ""
echo "All 4 models launched in parallel tmux sessions."
echo "Monitor with: ssh ${REMOTE_USER}@${REMOTE_HOST} 'tmux ls'"
echo "Attach to a session: ssh ${REMOTE_USER}@${REMOTE_HOST} 'tmux attach -t eval-phi4-mini'"
echo ""
echo "Wait for all sessions to complete, then run:"
echo "  bash run_full_experiment.sh --collect"

# ---------------------------------------------------------------------------
# Step 6: Collection mode (run with --collect flag)
# ---------------------------------------------------------------------------

if [ "${1:-}" = "--collect" ]; then
    echo ""
    echo "=========================================="
    echo "STEP 6: Running evaluation and analysis"
    echo "=========================================="

    remote_cmd "cd ${REMOTE_EVAL_DIR} && python3 -m src.evaluator --all \
        --output-dir ${REMOTE_EVAL_DIR} \
        --log-file ${REMOTE_EVAL_DIR}/evaluator.log"

    remote_cmd "cd ${REMOTE_EVAL_DIR} && python3 -m src.analyzer \
        --output-dir ${REMOTE_EVAL_DIR} \
        --log-file ${REMOTE_EVAL_DIR}/analyzer.log"

    echo ""
    echo "=========================================="
    echo "STEP 7: Collecting results"
    echo "=========================================="

    # Collect raw results
    echo "Downloading raw results..."
    remote_scp_from "${REMOTE_EVAL_DIR}/raw" "${LOCAL_EVAL_DIR}/"

    # Collect processed results
    echo "Downloading processed results..."
    remote_scp_from "${REMOTE_EVAL_DIR}/processed" "${LOCAL_EVAL_DIR}/"

    # Collect analysis
    echo "Downloading analysis..."
    remote_scp_from "${REMOTE_EVAL_DIR}/analysis" "${LOCAL_EVAL_DIR}/"

    # Collect logs
    echo "Downloading logs..."
    mkdir -p "${LOCAL_EVAL_DIR}/logs"
    remote_scp_from "${REMOTE_EVAL_DIR}/*.log" "${LOCAL_EVAL_DIR}/logs/" 2>/dev/null || true

    echo ""
    echo "=========================================="
    echo "EXPERIMENT COMPLETE"
    echo "=========================================="
    echo "Results at: ${LOCAL_EVAL_DIR}"
    echo "Key files:"
    echo "  ${LOCAL_EVAL_DIR}/analysis/comparison_table.csv"
    echo "  ${LOCAL_EVAL_DIR}/analysis/per_category_recall.csv"
    echo "  ${LOCAL_EVAL_DIR}/analysis/statistical_tests.json"
    echo "  ${LOCAL_EVAL_DIR}/analysis/pilot_results.json"
fi
