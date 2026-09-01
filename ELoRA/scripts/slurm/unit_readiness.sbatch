#!/usr/bin/env bash
#SBATCH --job-name=elora-unit
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
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
export MACE_CACHE_DIR=/home/xmz/symmetry-expert/.cache/mace
export MPLCONFIGDIR=/home/xmz/symmetry-expert/.cache/matplotlib
mkdir -p "${MACE_CACHE_DIR}" "${MPLCONFIGDIR}"
echo "job_id=${SLURM_JOB_ID} host=$(hostname) commit=${ACTUAL_COMMIT}"
python --version
CMD=(python -m pytest ELoRA/tests -q)
printf 'command:'; printf ' %q' "${CMD[@]}"; printf '\n'
srun --ntasks=1 "${CMD[@]}"
echo "end_time=$(date --iso-8601=seconds) status=success"
