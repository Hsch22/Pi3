#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PI3_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV_DIR="${PI3_VENV_DIR:-${PROJECT_ROOT}/.venv}"
MODEL_ID="${PI3_MODELSCOPE_MODEL:-Jasonhsc/Pi3}"
LOCAL_DIR="${PI3_MODELSCOPE_DIR:-${PROJECT_ROOT}/checkpoints/Jasonhsc/Pi3}"
CKPT_DIR="${PI3_CKPT_DIR:-${PROJECT_ROOT}/ckpts}"
WEIGHTS=(Pi3.safetensors Pi3X.safetensors)

fix_ownership() {
  if [ -n "${PI3_HOST_UID:-}" ] && [ -n "${PI3_HOST_GID:-}" ]; then
    chown -R "${PI3_HOST_UID}:${PI3_HOST_GID}" "${LOCAL_DIR}" "${CKPT_DIR}" 2>/dev/null || true
  fi
}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="*"
export no_proxy="*"

cd "${PROJECT_ROOT}"

if [ ! -x "${VENV_DIR}/bin/modelscope" ]; then
  bash scripts/bootstrap_musa_uv.sh
fi

mkdir -p "${LOCAL_DIR}" "${CKPT_DIR}"

if [ "${PI3_FORCE_DOWNLOAD:-0}" != "1" ]; then
  ready=1
  for weight in "${WEIGHTS[@]}"; do
    if [ ! -s "${CKPT_DIR}/${weight}" ]; then
      ready=0
    fi
  done
  if [ "${ready}" = "1" ]; then
    fix_ownership
    echo "Pi3 checkpoints already ready:"
    for weight in "${WEIGHTS[@]}"; do
      echo "  ${CKPT_DIR}/${weight}"
    done
    exit 0
  fi
fi

if [ -n "${MODELSCOPE_TOKEN:-}" ]; then
  "${VENV_DIR}/bin/modelscope" login --token "${MODELSCOPE_TOKEN}"
else
  echo "MODELSCOPE_TOKEN is not set; using existing ModelScope credentials if available."
fi

"${VENV_DIR}/bin/modelscope" download --model "${MODEL_ID}" --local_dir "${LOCAL_DIR}"

for weight in "${WEIGHTS[@]}"; do
  if [ -s "${LOCAL_DIR}/${weight}" ]; then
    mv -f "${LOCAL_DIR}/${weight}" "${CKPT_DIR}/${weight}"
  fi
done

fix_ownership

echo "Pi3 checkpoints ready:"
for weight in "${WEIGHTS[@]}"; do
  echo "  ${CKPT_DIR}/${weight}"
done
