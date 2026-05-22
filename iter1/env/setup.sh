#!/usr/bin/env bash
# iter1 setup — Mac CPU (M-series Apple Silicon). No CUDA wheels.
# Per protocol §"Dependencies". Do not edit without updating protocol.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 1. Python check — protocol assumes 3.10+
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python: $python_version"
case "$python_version" in
    3.10|3.11|3.12) ;;
    *) echo "WARN: protocol tested against Python 3.10-3.12; you have $python_version" ;;
esac

# 2. Force CPU-only torch on M-series — pulling cuda wheels here will fail and waste bandwidth.
echo "==> Installing CPU torch first to avoid CUDA wheel pull"
python3 -m pip install --upgrade pip
python3 -m pip install "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu || \
    python3 -m pip install "torch>=2.0.0"

# 3. Box2D needs SWIG on M-series; install it first.
echo "==> Installing swig (Box2D dependency)"
if command -v brew >/dev/null 2>&1; then
    brew list swig >/dev/null 2>&1 || brew install swig
else
    echo "WARN: brew not found; skipping swig. BipedalWalker install may fail."
fi
python3 -m pip install swig || true

# 4. Main requirements
echo "==> Installing project requirements"
python3 -m pip install -r env/requirements.txt

# 5. Verify Box2D actually works. Protocol §"Setup phase" says drop BipedalWalker
#    and document if this fails.
echo "==> Verifying Box2D import"
if python3 -c "import gymnasium; gymnasium.make('BipedalWalker-v3').close()" 2>/dev/null; then
    echo "    Box2D OK — BipedalWalker available."
else
    echo "    WARN: Box2D / BipedalWalker not available. Per protocol you may need to"
    echo "    reduce scope to Pendulum + MountainCar and document in critique.md."
fi

echo ""
echo "Setup complete. Next: python env/verify_env.py"
