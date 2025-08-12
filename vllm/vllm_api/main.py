import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from controller.vllmController import router as vllm_router, auto_serve_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("polar-trainer")

app = FastAPI(
    title="Polar vLLM API",
    description="API for training models with customizable parameters",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(vllm_router)

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행되는 이벤트"""
    auto_serve_enabled = os.getenv("AUTO_SERVE_MODEL", "false").lower() == "true"
    model_name = os.getenv("VLLM_MODEL_NAME", "")

    if auto_serve_enabled and model_name:
        logger.info("자동 모델 서빙이 활성화되었습니다. 모델: %s", model_name)
        try:
            # 백그라운드에서 모델 서빙 시작
            asyncio.create_task(auto_serve_model(model_name))
        except Exception as e:
            logger.error("자동 모델 서빙 중 오류 발생: %s", str(e))
    else:
        logger.info("자동 모델 서빙이 비활성화되었습니다.")

if __name__ == "__main__":
    # Remove reload=True to prevent automatic restarts
    port = int(os.getenv("VLLM_CONTROLLER_PORT", "12435"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
