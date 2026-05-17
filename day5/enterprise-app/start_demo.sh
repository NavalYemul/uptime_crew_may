#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
UI_HOST="${UI_HOST:-127.0.0.1}"
UI_PORT="${UI_PORT:-8501}"
BACKEND_URL="http://${BACKEND_HOST}:${BACKEND_PORT}"

if [[ ! -d ".venv" ]]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

if [[ ! -f ".env" ]]; then
  echo "Missing .env file."
  echo "Create one from .env.example and add your OpenAI API key:"
  echo "  cp .env.example .env"
  exit 1
fi

set -a
source .env
set +a

if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "your_openai_api_key_here" ]]; then
  echo "OPENAI_API_KEY is missing in .env"
  exit 1
fi

if ! .venv/bin/python -c "import fastapi, streamlit, langchain_openai, pypdf, ddgs" >/dev/null 2>&1; then
  echo "Installing dependencies..."
  .venv/bin/pip install -r requirements.txt
fi

echo "Starting backend on ${BACKEND_URL}"
.venv/bin/python -m uvicorn app:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" &
BACKEND_PID=$!

cleanup() {
  kill "${BACKEND_PID}" >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

sleep 2

echo "Starting demo UI on http://${UI_HOST}:${UI_PORT}"
BACKEND_URL="${BACKEND_URL}" .venv/bin/python -m streamlit run ui.py --server.address "${UI_HOST}" --server.port "${UI_PORT}"
