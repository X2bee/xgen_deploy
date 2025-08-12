#!/bin/bash
cd /home/xgen_front

cd /home/xgen_front
set -e
echo "📥 Installing dependencies..."
npm install

echo "✅ Setup complete"
echo "🚀 Building the app..."
if ! npm run build; then
  echo "앱 빌드 실패. 컨테이너를 유지하기 위해 대기 모드로 전환합니다..."
  tail -f /dev/null
else
  echo "✅ Build successful"
  echo "🚀 Starting the server..."
  npm run start
fi
