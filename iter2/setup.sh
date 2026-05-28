#!/usr/bin/env bash
# iter2 setup — creates .venv and installs deps.
# Python 3.10+ required (mujoco wheels only ship for >=3.10). Tested on 3.14.
set -euo pipefail

cd "$(dirname "$0")"

# Resolve a Python interpreter. Prefer an explicit override, then any modern
# homebrew/system Python, falling back to whatever `python3` points at.
choose_python() {
  if [ -n "${PYTHON:-}" ]; then
    echo "$PYTHON"; return
  fi
  for cand in python3.12 python3.13 python3.14 python3.11 /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      ver=$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
      major=${ver%%.*}
      minor=${ver##*.}
      if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
        echo "$cand"; return
      fi
    fi
  done
  echo ""
}

PY="$(choose_python)"
if [ -z "$PY" ]; then
  echo "[error] no Python >= 3.10 found. Install one (e.g. 'brew install python@3.12') and retry."
  exit 1
fi

echo "[setup] using $PY ($("$PY" --version))"

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt

echo
echo "[setup] done. Activate with:  source iter2/.venv/bin/activate"
echo "[setup] verify with:          python -m src.verify"
