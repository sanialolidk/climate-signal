#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate 2>/dev/null || true
pip install -q -r requirements-api.txt
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000