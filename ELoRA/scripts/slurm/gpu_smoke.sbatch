#!/usr/bin/env bash
#SBATCH --job-name=elora-gpu-smoke
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/xmz/symmetry-expert/slurm-%x-%j.out
#SBATCH --error=/home/xmz/symmetry-expert/slurm-%x-%j.err
#SBATCH --signal=B:TERM@60

set -Eeuo pipefail
trap 'status=$?; echo "[error] exit_code=${status} host=$(hostname) time=$(date --iso-8601=seconds)" >&2; exit "${status}"' ERR
: "${EXPECTED_COMMIT:?submit with EXPECTED_COMMIT set}"
PROJECT_DIR=/home/xmz/symmetry-expert
cd "${PROJECT_DIR}"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
[[ "${ACTUAL_COMMIT}" == "${EXPECTED_COMMIT}" ]] || { echo "commit mismatch expected=${EXPECTED_COMMIT} actual=${ACTUAL_COMMIT}" >&2; exit 2; }
source /home/xmz/symmetry-expert/.venv/bin/activate
echo "job_id=${SLURM_JOB_ID} host=$(hostname) commit=${ACTUAL_COMMIT} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
for UPDATE_MODE in dense elora_clean elora_paper; do
    OUTPUT_MODE="${UPDATE_MODE//_/-}"
    CMD=(python ELoRA/scripts/readiness_smoke.py --device cuda --update-mode "${UPDATE_MODE}" --output "ELoRA/artifacts/readiness/gpu-smoke-${OUTPUT_MODE}.json")
    printf 'command:'; printf ' %q' "${CMD[@]}"; printf '\n'
    srun --ntasks=1 "${CMD[@]}"
done
echo "end_time=$(date --iso-8601=seconds) status=success"
