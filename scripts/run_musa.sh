#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PI3_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
IMAGE_TAG="${PI3_IMAGE:-registry.mthreads.com/mcctest/ai/mtwan:4.3.3-pt2.7-v0.2-mudnn3.1.7-ph1}"
SHARED_IMAGE_LOADER="${PI3_SHARED_IMAGE_LOADER:-/datapool/shared_images/mtwan_4.3.3-pt2.7-v0.2-mudnn3.1.7-ph1/load_mtwan.sh}"
PROXY_ENV="${PI3_PROXY_ENV:-/datapool/.config/mihomo/proxy-env.sh}"
CROCO_MUSA_PATH="${PI3_CROCO_MUSA_PATH:-${PROJECT_ROOT}/third_party/croco_musa}"

if [ "$#" -eq 0 ]; then
  set -- bash
fi

if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  bash "${SHARED_IMAGE_LOADER}"
fi

docker run --rm -i \
  --privileged \
  --network=host \
  --ipc=host \
  --runtime mthreads \
  -v /datapool:/datapool \
  -w "${PROJECT_ROOT}" \
  -e PI3_BOOTSTRAP_IN_CONTAINER=1 \
  -e PI3_PROJECT_ROOT="${PROJECT_ROOT}" \
  -e PI3_VENV_DIR="${PI3_VENV_DIR:-${PROJECT_ROOT}/.venv}" \
  -e PI3_PYPI_MIRROR="${PI3_PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
  -e PI3_PYPI_FALLBACK="${PI3_PYPI_FALLBACK:-https://pypi.org/simple}" \
  -e PI3_PROXY_ENV="${PROXY_ENV}" \
  -e PI3_CROCO_MUSA_PATH="${CROCO_MUSA_PATH}" \
  -e PI3_ENABLE_CROCO_MUSA="${PI3_ENABLE_CROCO_MUSA:-1}" \
  -e PI3_INSTALL_DEMO="${PI3_INSTALL_DEMO:-0}" \
  -e MODELSCOPE_TOKEN="${MODELSCOPE_TOKEN:-}" \
  -e UV_LINK_MODE="${UV_LINK_MODE:-copy}" \
  -e MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0}" \
  -e PI3_HOST_UID="$(id -u)" \
  -e PI3_HOST_GID="$(id -g)" \
  "${IMAGE_TAG}" \
  bash -lc '
    set -euo pipefail
    if [ "${PI3_ENABLE_CROCO_MUSA:-1}" = "1" ] && [ -d "${PI3_CROCO_MUSA_PATH}" ]; then
      export PYTHONPATH="${PI3_CROCO_MUSA_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
    fi
    bash scripts/bootstrap_musa_uv.sh
    if [ "${PI3_ENABLE_CROCO_MUSA:-1}" = "1" ] && [ -d "${PI3_CROCO_MUSA_PATH}" ]; then
      CUROPE_DIR="${PI3_CROCO_MUSA_PATH}/models/curope"
      CUROPE_SO=""
      for path in "${CUROPE_DIR}"/curope*.so; do
        if [ -e "$path" ]; then
          CUROPE_SO="$path"
          break
        fi
      done
      CUROPE_NEEDS_BUILD=0
      if [ -z "${CUROPE_SO}" ]; then
        CUROPE_NEEDS_BUILD=1
      elif [ "${CUROPE_DIR}/setup.py" -nt "${CUROPE_SO}" ] || \
           [ "${CUROPE_DIR}/curope.cpp" -nt "${CUROPE_SO}" ] || \
           [ "${CUROPE_DIR}/kernels.mu" -nt "${CUROPE_SO}" ]; then
        CUROPE_NEEDS_BUILD=1
      fi
      if [ "${CUROPE_NEEDS_BUILD}" = "1" ]; then
        (cd "${CUROPE_DIR}" && "${PI3_VENV_DIR}/bin/python" setup.py build_ext --inplace)
        chown -R "${PI3_HOST_UID}:${PI3_HOST_GID}" \
          "${CUROPE_DIR}/build" \
          "${CUROPE_DIR}"/curope*.so 2>/dev/null || true
      fi
    fi
    exec "$@"
  ' bash "$@"
