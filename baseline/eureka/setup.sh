#!/usr/bin/env bash
# Eureka baseline — Mac M-series CPU. No CUDA wheels.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python: $python_version"
case "$python_version" in
    3.10|3.11|3.12) ;;
    *) echo "WARN: tested against 3.10-3.12; you have $python_version" ;;
esac

python3 -m pip install --upgrade pip
python3 -m pip install "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu || \
    python3 -m pip install "torch>=2.0.0"

python3 -m pip install -r requirements.txt

echo "==> Verifying MuJoCo import + HalfCheetah-v4"
if python3 -c "import gymnasium as g; g.make('HalfCheetah-v4').close()"; then
    echo "    OK"
else
    echo "    FAILED — fix MuJoCo install before running eureka.py"
    exit 1
fi

echo ""
echo "Setup complete. Set ANTHROPIC_API_KEY, then: python verify_env.py"
