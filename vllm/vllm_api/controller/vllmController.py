import os
import time
import subprocess
import signal
import platform
import psutil
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Any, List
from huggingface_hub import model_info, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

router = APIRouter(
    prefix="/api/vllm",
    tags=["vllm"],
    responses={404: {"description": "Not found"}},
)

# vLLM 프로세스 객체를 저장할 전역 변수
vllm_process: Optional[subprocess.Popen] = None

async def auto_serve_model(model_name: str):
    try:
        model_config = ModelConfig(
            model_id=model_name,
            tokenizer=os.getenv("VLLM_TOKENIZER", model_name),
            download_dir=os.getenv("VLLM_DOWNLOAD_DIR", "/models/huggingface"),
            host=os.getenv("VLLM_HOST", "0.0.0.0"),
            port=int(os.getenv("VLLM_PORT", "12434")),
            max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "32768")),
            pipeline_parallel_size=int(os.getenv("VLLM_PIPELINE_PARALLEL_SIZE", "1")),
            tensor_parallel_size=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
            gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.95")),
            dtype=os.getenv("VLLM_DTYPE", "bfloat16"),
            kv_cache_dtype=os.getenv("VLLM_KV_CACHE_DTYPE", "auto"),
            load_local=os.getenv("VLLM_LOAD_LOCAL", "false").lower() == "true",
            tool_call_parser=os.getenv("VLLM_TOOL_CALL_PARSER", None)
        )

        # 약간의 지연 후 모델 서빙 시작
        await asyncio.sleep(5)
        print(f"자동 모델 서빙을 시작합니다: {model_name}")
        await vllm_serve_model(model_config)

    except Exception as e:
        print(f"자동 모델 서빙 실패: {e}")

def _terminate_process_safely(process: subprocess.Popen) -> bool:
    """
    크로스 플랫폼 호환 프로세스 안전 종료 함수
    """
    if process is None or process.poll() is not None:
        return True

    try:
        pid = process.pid
        print(f"Attempting to terminate process with PID: {pid}")

        if platform.system() == "Windows":
            # Windows에서는 psutil을 사용하여 프로세스 트리 종료
            try:
                parent = psutil.Process(pid)
                children = parent.children(recursive=True)

                # 자식 프로세스들을 먼저 종료
                for child in children:
                    try:
                        print(f"Terminating child process: {child.pid}")
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass

                # 부모 프로세스 종료
                parent.terminate()

                # 정상 종료 대기 (10초)
                try:
                    parent.wait(timeout=10)
                    print(f"Process {pid} terminated gracefully")
                    return True
                except psutil.TimeoutExpired:
                    print(f"Process {pid} didn't terminate gracefully, forcing kill")
                    # 강제 종료
                    for child in children:
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    parent.kill()
                    parent.wait(timeout=5)
                    print(f"Process {pid} forcefully killed")
                    return True

            except psutil.NoSuchProcess:
                print(f"Process {pid} not found")
                return True
            except Exception as e:
                print(f"Error terminating Windows process: {e}")
                return False

        else:
            # Unix/Linux에서는 기존 방식 사용
            try:
                if hasattr(os, 'getpgid') and hasattr(os, 'killpg'):
                    pgid = os.getpgid(pid)
                    print(f"Terminating process group with PGID: {pgid}")
                    os.killpg(pgid, signal.SIGTERM)

                    try:
                        process.wait(timeout=10)
                        print(f"Process group {pgid} terminated gracefully")
                        return True
                    except subprocess.TimeoutExpired:
                        print(f"Process group {pgid} didn't terminate gracefully, forcing kill")
                        if hasattr(signal, 'SIGKILL'):
                            os.killpg(pgid, signal.SIGKILL)
                        else:
                            os.killpg(pgid, signal.SIGTERM)
                        process.wait(timeout=5)
                        print(f"Process group {pgid} forcefully killed")
                        return True
                else:
                    # 기본적인 프로세스 종료
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                        return True
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                        return True

            except ProcessLookupError:
                print(f"Process {pid} not found")
                return True
            except Exception as e:
                print(f"Error terminating Unix process: {e}")
                return False

    except Exception as e:
        print(f"Unexpected error in _terminate_process_safely: {e}")
        return False

class HFConfig(BaseModel):
    model_id: str = Field(..., description="Model ID")
    download_dir: Optional[str] = Field("/models/huggingface", description="Model download directory")
    allow_patterns: Optional[List[str]] = Field(None, description="Model download patterns")

class ModelConfig(BaseModel):
    model_id: str = Field(..., description="Model ID")
    tokenizer: Optional[str] = Field(None, description="Tokenizer")
    download_dir: Optional[str] = Field("/models/huggingface", description="Model download directory")
    host: Optional[str] = Field("0.0.0.0", description="Host address")
    port: Optional[int] = Field(12434, description="Port number")
    max_model_len: Optional[int] = Field(8192, description="Maximum model length")
    pipeline_parallel_size: Optional[int] = Field(1, description="Pipeline parallel size")
    tensor_parallel_size: Optional[int] = Field(1, description="Tensor parallel size")
    gpu_memory_utilization: Optional[float] = Field(0.95, description="GPU memory utilization")
    dtype: Optional[str] = Field('bfloat16', description="Data type {auto,half,float16,bfloat16,float,float32}")
    kv_cache_dtype: Optional[str] = Field('auto', description="KV cache data type {auto,fp8}")
    load_local: Optional[bool] = Field(False, description="Whether to load models from local drive")
    tool_call_parser: Optional[str] = Field(None, description="Tool call parser")

class SuccessResponse(BaseModel):
    status: str = "success"
    message: str
    data: Optional[Any] = None

@router.post("/hf/health", response_model=SuccessResponse)
async def hf_health_check(params: HFConfig):
    try:
        model_info(params.model_id)
        return SuccessResponse(
            message=f"Model {params.model_id} is available on Hugging Face Hub.",
            data={
                "model_id": params.model_id,
                "exists": True,
            }
        )
    except RepositoryNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model {params.model_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hf/download", response_model=SuccessResponse)
async def hf_repo_download(params: HFConfig):
    try:
        model_id_path = (params.model_id).replace("/", "__")
        model_subdir = os.path.join(params.download_dir, model_id_path)
        os.makedirs(model_subdir, exist_ok=True)

        effective_allow_patterns = params.allow_patterns
        if isinstance(effective_allow_patterns, str):
            effective_allow_patterns = [pattern.strip() for pattern in effective_allow_patterns.split(",")]

        print(f"[INFO] Download Start ... Model ID: {params.model_id}, Download Directory: {model_subdir}")
        snapshot_download(
            params.model_id,
            local_dir=model_subdir,
            allow_patterns=effective_allow_patterns
        )
        return SuccessResponse(
            message=f"[INFO] All files for repo {params.model_id} have been downloaded to {model_subdir}.",
            data={
                "model_id": params.model_id,
                "download_dir": model_subdir,
            }
        )
    except RepositoryNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model {params.model_id} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/serve", response_model=SuccessResponse)
async def vllm_serve_model(params: ModelConfig):
    global vllm_process

    # 이미 다른 vLLM 프로세스가 실행 중인지 확인
    if vllm_process and vllm_process.poll() is None:
        raise HTTPException(status_code=400, detail="A vLLM model is already being served. Please shut it down first.")

    revised_model_id = params.model_id.replace("/", "__")
    model_path = os.path.join(params.download_dir, revised_model_id)

    if params.load_local and not os.path.exists(model_path):
        await hf_repo_download(HFConfig(model_id=params.model_id, download_dir=params.download_dir))

    if params.tokenizer == 'string' or params.tokenizer == "":
        params.tokenizer = params.model_id

    command_parts = [
        "vllm", "serve", params.model_id,
        "--host", params.host,
        "--port", str(params.port),
        "--trust-remote-code",
        "--max-model-len", str(params.max_model_len),
        "--pipeline-parallel-size", str(params.pipeline_parallel_size),
        "--tensor-parallel-size", str(params.tensor_parallel_size),
        "--gpu-memory-utilization", str(params.gpu_memory_utilization),
        "--dtype", params.dtype,
        "--kv-cache-dtype", params.kv_cache_dtype,
        "--enable-auto-tool-choice",
    ]
    if params.tokenizer:
        command_parts.extend(["--tokenizer", params.tokenizer])

    if params.tool_call_parser:
        command_parts.extend(["--tool-call-parser", params.tool_call_parser])

    if not params.load_local:
        command_parts.extend(["--download-dir", params.download_dir])

    try:
        print("vLLM command to be executed:")
        print(" ".join(command_parts))

        # Windows와 Unix 호환성을 위한 프로세스 시작
        if platform.system() == "Windows":
            # Windows에서는 새로운 프로세스 그룹을 생성
            vllm_process = subprocess.Popen(
                command_parts,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            # Unix/Linux에서는 새로운 세션을 생성
            if hasattr(os, 'setsid'):
                vllm_process = subprocess.Popen(
                    command_parts,
                    preexec_fn=os.setsid
                )
            else:
                vllm_process = subprocess.Popen(command_parts)

        time.sleep(10)

        return SuccessResponse(
            message=f"Model '{params.model_id}' serve initiated successfully with PID: {vllm_process.pid}.",
            data={"pid": vllm_process.pid}
        )

    except Exception as e:
        if 'vllm_process' in locals() and vllm_process and vllm_process.poll() is None:
            try:
                # 안전한 프로세스 종료
                _terminate_process_safely(vllm_process)
            except:  # noqa: E722
                pass
        vllm_process = None
        raise HTTPException(status_code=500, detail=f"Failed to start vLLM process: {str(e)}")

@router.post("/down", response_model=SuccessResponse)
async def vllm_down_model():
    global vllm_process

    if vllm_process is None:
        raise HTTPException(status_code=404, detail="No vLLM model process found to shut down.")

    # 프로세스가 이미 종료되었는지 확인
    if vllm_process.poll() is not None:
        vllm_process = None
        raise HTTPException(status_code=404, detail="vLLM model process is already terminated.")

    pid = vllm_process.pid
    print(f"Shutting down vLLM model process with PID: {pid}")

    try:
        # 안전한 프로세스 종료 시도
        success = _terminate_process_safely(vllm_process)

        if success:
            message = f"vLLM model process with PID {pid} has been successfully shut down."
            print(message)
        else:
            # 종료에 실패한 경우에도 전역 변수는 초기화하고 경고 메시지 반환
            message = f"vLLM model process with PID {pid} may not have been cleanly shut down. Please check manually."
            print(f"WARNING: {message}")

        # 전역 변수 초기화
        vllm_process = None

        return SuccessResponse(message=message)

    except Exception as e:
        # 예상치 못한 오류 발생 시에도 전역 변수는 초기화
        print(f"Error during process shutdown: {str(e)}")
        vllm_process = None
        raise HTTPException(status_code=500, detail=f"An error occurred while shutting down the process: {str(e)}")

@router.get("/health", response_model=SuccessResponse)
async def vllm_health():
    """vLLM 컨트롤러 헬스 체크 API"""
    global vllm_process

    # 기본 헬스 정보
    health_data = {
        "controller_status": "healthy",
        "timestamp": time.time(),
        "platform": platform.system(),
        "process_info": None
    }

    # vLLM 프로세스 상태 정보 추가
    if vllm_process is not None:
        poll_result = vllm_process.poll()
        if poll_result is None:
            # 프로세스가 실행 중
            health_data["process_info"] = {
                "status": "running",
                "pid": vllm_process.pid,
                "running": True
            }
        else:
            # 프로세스가 종료됨
            health_data["process_info"] = {
                "status": "terminated",
                "pid": None,
                "running": False,
                "exit_code": poll_result
            }
            # 전역 변수 정리
            vllm_process = None
    else:
        health_data["process_info"] = {
            "status": "not_running",
            "pid": None,
            "running": False
        }

    return SuccessResponse(
        message="vLLM Controller is healthy and operational",
        data=health_data
    )
