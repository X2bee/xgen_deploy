#!/bin/bash
set -e

cd /plateerag_backend

pip install uv

uv sync

if ! uv run main.py; then
  echo "main.py 실행 실패. 컨테이너를 유지하기 위해 대기 모드로 전환합니다..."
  tail -f /dev/null
fi
