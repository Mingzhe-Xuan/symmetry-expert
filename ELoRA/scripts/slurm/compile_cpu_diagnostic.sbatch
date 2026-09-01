#!/usr/bin/env bash
#SBATCH --job-name=elora-compile-diag
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/xmz/symmetry-expert/slurm-%x-%j.out
#SBATCH --error=/home/xmz/symmetry-expert/slurm-%x-%j.err

set -Eeuo pipefail
trap 'status=$?; echo "[error] exit_code=${status} host=$(hostname) time=$(date --iso-8601=seconds)" >&2; exit "${status}"' ERR
: "${EXPECTED_COMMIT:?submit with EXPECTED_COMMIT set}"
: "${DIAG_PROFILE:?fresh_cache or fresh_cache_single_thread}"

PROJECT_DIR=/home/xmz/symmetry-expert
cd "${PROJECT_DIR}"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
[[ "${ACTUAL_COMMIT}" == "${EXPECTED_COMMIT}" ]] || { echo "commit mismatch expected=${EXPECTED_COMMIT} actual=${ACTUAL_COMMIT}" >&2; exit 2; }
source /home/xmz/symmetry-expert/.venv/bin/activate

export TORCHINDUCTOR_CACHE_DIR="${PROJECT_DIR}/.cache/torchinductor/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${PROJECT_DIR}/.cache/triton/${SLURM_JOB_ID}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

case "${DIAG_PROFILE}" in
    fresh_cache)
        ;;
    fresh_cache_single_thread)
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export TORCHINDUCTOR_COMPILE_THREADS=1
        ;;
    *)
        echo "unsupported DIAG_PROFILE=${DIAG_PROFILE}" >&2
        exit 2
        ;;
esac

echo "job_id=${SLURM_JOB_ID} host=$(hostname) commit=${ACTUAL_COMMIT} profile=${DIAG_PROFILE}"
python --version
CMD=(python -m pytest \
    'ELoRA/tests/test_compile.py::test_mace[fp32-cpu]' \
    'ELoRA/tests/test_compile.py::test_mace[fp64-cpu]' \
    -q -vv)
printf 'command:'; printf ' %q' "${CMD[@]}"; printf '\n'
srun --ntasks=1 "${CMD[@]}"
echo "end_time=$(date --iso-8601=seconds) status=success"
