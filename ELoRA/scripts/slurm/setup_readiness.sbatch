#!/usr/bin/env bash
#SBATCH --job-name=elora-env
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/xmz/symmetry-expert/slurm-%x-%j.out
#SBATCH --error=/home/xmz/symmetry-expert/slurm-%x-%j.err
#SBATCH --signal=B:TERM@60

set -Eeuo pipefail
trap 'status=$?; echo "[error] exit_code=${status} host=$(hostname) time=$(date --iso-8601=seconds)" >&2; exit "${status}"' ERR

: "${EXPECTED_COMMIT:?submit with EXPECTED_COMMIT set}"
PROJECT_DIR=/home/xmz/symmetry-expert
VENV_DIR=/home/xmz/symmetry-expert/.venv
WHEELHOUSE=/home/xmz/symmetry-expert/.cache/wheelhouse
PIP_CACHE_DIR=/home/xmz/symmetry-expert/.cache/pip
export PIP_CACHE_DIR

cd "${PROJECT_DIR}"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
[[ "${ACTUAL_COMMIT}" == "${EXPECTED_COMMIT}" ]] || { echo "commit mismatch expected=${EXPECTED_COMMIT} actual=${ACTUAL_COMMIT}" >&2; exit 2; }
echo "job_id=${SLURM_JOB_ID} host=$(hostname) commit=${ACTUAL_COMMIT}"
echo "start_time=$(date --iso-8601=seconds)"
TORCH_PARTS="${WHEELHOUSE}/torch.parts"
TORCH_WHEEL="${WHEELHOUSE}/torch-2.11.0+cu128-cp310-cp310-manylinux_2_28_x86_64.whl"
if [[ -f "${TORCH_PARTS}/SHA256SUMS" ]] && (cd "${TORCH_PARTS}" && sha256sum --check SHA256SUMS); then
    echo "torch_parts=verified; reconstructing wheel"
    cat "${TORCH_PARTS}"/part-* > "${TORCH_WHEEL}.tmp"
    mv -f "${TORCH_WHEEL}.tmp" "${TORCH_WHEEL}"
elif [[ -d "${TORCH_PARTS}" ]]; then
    echo "torch_parts=incomplete; skipping reconstruction"
fi

USE_OFFLINE=0
if [[ -f "${WHEELHOUSE}/SHA256SUMS" ]] && (cd "${WHEELHOUSE}" && sha256sum --check SHA256SUMS); then
    USE_OFFLINE=1
    echo "dependency_source=verified_offline_wheelhouse"
else
    echo "dependency_source=online_pypi_and_pytorch_cu128"
fi

/usr/bin/python3 -m venv --clear "${VENV_DIR}"
if [[ "${USE_OFFLINE}" == "1" ]]; then
    "${VENV_DIR}/bin/python" -m pip install --no-index --find-links "${WHEELHOUSE}" --upgrade pip setuptools wheel
    "${VENV_DIR}/bin/python" -m pip install --no-index --find-links "${WHEELHOUSE}" -r docs/requirements.txt
else
    PIP_NETWORK_ARGS=(--timeout 120 --retries 20)
    "${VENV_DIR}/bin/python" -m pip install "${PIP_NETWORK_ARGS[@]}" --upgrade pip setuptools wheel
    "${VENV_DIR}/bin/python" -m pip install "${PIP_NETWORK_ARGS[@]}" -r docs/requirements.txt
fi
"${VENV_DIR}/bin/python" -m pip install --no-index --no-deps -e ELoRA
"${VENV_DIR}/bin/python" -m pip check
"${VENV_DIR}/bin/python" - <<'PY'
import ase, e3nn, h5py, matplotlib, numpy, pandas, scipy, torch
print("python_environment=ok")
print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("e3nn=", e3nn.__version__)
print("ase=", ase.__version__)
print("numpy=", numpy.__version__)
print("scipy=", scipy.__version__)
print("h5py=", h5py.__version__)
print("matplotlib=", matplotlib.__version__)
print("pandas=", pandas.__version__)
PY
echo "end_time=$(date --iso-8601=seconds) status=success"
