#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
ROCM_VERSION="7.2.4"
ROCM_BUILD="70204"
WIN_SDK="/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0"
WORK_DIR="$(mktemp -d)"

cleanup() {
    rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

if ! grep -qi microsoft /proc/version; then
    echo "Erro: este instalador deve ser executado dentro do WSL2."
    exit 1
fi

if [[ ! -e /dev/dxg ]]; then
    cat <<'EOF'
Erro: /dev/dxg nao existe. A GPU ainda nao foi exposta ao WSL2.

No PowerShell do Windows, execute:
  wsl --shutdown

Abra novamente o Ubuntu e rode:
  ls -l /dev/dxg

Se o arquivo continuar ausente, instale o driver AMD Adrenalin com suporte
a WSL2 e reinicie o Windows antes de executar este script novamente.
EOF
    exit 2
fi

if [[ ! -d "${WIN_SDK}/shared" ]]; then
    echo "Erro: Windows SDK nao encontrado em ${WIN_SDK}."
    exit 3
fi

echo "Instalando dependencias de sistema e ROCm ${ROCM_VERSION}..."
sudo apt update
sudo apt install -y wget git cmake build-essential python3-setuptools python3-wheel

cd "${WORK_DIR}"
wget "https://repo.radeon.com/amdgpu-install/${ROCM_VERSION}/ubuntu/noble/amdgpu-install_${ROCM_VERSION}.${ROCM_BUILD}-1_all.deb"
sudo apt install -y "./amdgpu-install_${ROCM_VERSION}.${ROCM_BUILD}-1_all.deb"
sudo apt update
sudo apt install -y rocm

if getent group render >/dev/null && getent group video >/dev/null; then
    sudo usermod -a -G render,video "${USER}"
fi

echo "Compilando a integracao ROCDXG..."
git clone --depth 1 --branch develop https://github.com/ROCm/librocdxg.git
cmake -S librocdxg -B librocdxg/build -DWIN_SDK="${WIN_SDK}/shared"
cmake --build librocdxg/build --parallel
sudo cmake --install librocdxg/build

export HSA_ENABLE_DXG_DETECTION=1
export PATH="/opt/rocm/bin:${PATH}"

echo "Validando a deteccao da GPU pelo ROCm..."
rocminfo | grep -E "Name:|Marketing Name:" | head -20

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    python3 -m venv "${VENV_DIR}"
fi

PYTHON_VERSION="$("${VENV_DIR}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_VERSION}" != "3.12" ]]; then
    echo "Erro: TensorFlow ROCm requer Python 3.12; a venv usa ${PYTHON_VERSION}."
    exit 4
fi

echo "Instalando TensorFlow ROCm na venv..."
"${VENV_DIR}/bin/python" -m pip uninstall -y tensorflow tensorflow-cpu tensorflow-rocm || true
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${PROJECT_DIR}/requirements-rocm.txt"

echo "Validando o TensorFlow..."
HSA_ENABLE_DXG_DETECTION=1 "${VENV_DIR}/bin/python" - <<'PY'
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow:", tf.__version__)
print("GPUs:", gpus)
if not gpus:
    raise SystemExit("Erro: TensorFlow ainda nao detectou a GPU.")
PY

cat <<'EOF'

Configuracao concluida.
Reabra o VS Code, selecione o kernel da .venv e execute o notebook novamente.
EOF
