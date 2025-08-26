#!/bin/bash
set -e

cd /vllm/vllm_api
echo "필수 패키지를 설치합니다..."
pip install -r requirements.txt

if [ -n "$VLLM_MODEL_NAME" ] && [ "$VLLM_MODEL_NAME" != "" ]; then
    echo "VLLM_MODEL_NAME이 설정되어 있습니다: $VLLM_MODEL_NAME"
    echo "자동 모델 서빙 모드로 실행합니다..."
    export AUTO_SERVE_MODEL=true
else
    echo "VLLM_MODEL_NAME이 설정되지 않았습니다. 수동 서빙 모드로 실행합니다..."
    export AUTO_SERVE_MODEL=false
fi

if ! python3 main.py; then
  echo "앱 빌드 실패. 컨테이너를 유지하기 위해 대기 모드로 전환합니다..."
  tail -f /dev/null
fi
