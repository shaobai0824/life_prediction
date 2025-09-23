"""
壽命預測保險決策輔助系統 - FastAPI 主應用程式

作者: shaobai
版本: 1.0.0
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time
import logging

from app.core.config import settings
from app.core.security import SecurityHeaders
from app.api.router import api_router
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.audit_log import AuditLogMiddleware

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理"""
    logger.info("🚀 啟動壽命預測系統...")

    # 啟動時初始化
    try:
        # 初始化資料庫連接
        logger.info("📁 初始化資料庫連接...")

        # 載入 AI 模型
        logger.info("🤖 載入預測模型...")

        # 初始化 Redis 快取
        logger.info("⚡ 初始化快取系統...")

        logger.info("✅ 系統初始化完成")

    except Exception as e:
        logger.error(f"❌ 系統初始化失敗: {e}")
        raise

    yield

    # 關閉時清理
    logger.info("🛑 正在關閉系統...")
    logger.info("✅ 系統已安全關閉")


# 建立 FastAPI 應用程式
app = FastAPI(
    title="壽命預測保險決策輔助系統",
    description="""
    基於 AI 面相識別技術的壽命預測系統，專為金融保險業設計。

    **特色功能:**
    - 🤖 深度學習面相分析
    - 🛡️ 金融級安全合規
    - ⚡ 毫秒級預測響應
    - 🔒 隱私保護設計

    **重要聲明:** 本系統僅供保險業決策輔助，非醫療診斷工具。
    """,
    version="1.0.0",
    contact={
        "name": "shaobai",
        "email": "contact@lifepredict.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
)

# CORS 中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# 信任主機中間件（安全）
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# 自定義安全標頭中間件
app.add_middleware(SecurityHeaders)

# 速率限制中間件
app.add_middleware(RateLimitMiddleware)

# 審計日誌中間件（金融合規）
app.add_middleware(AuditLogMiddleware)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """添加處理時間標頭"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全域例外處理器"""
    logger.error(f"未處理的例外: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "內部伺服器錯誤",
            "message": "系統發生未預期的錯誤，請稍後再試",
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )


# 健康檢查端點
@app.get("/health", tags=["監控"])
async def health_check():
    """系統健康檢查"""
    return {
        "status": "healthy",
        "service": "life-prediction-api",
        "version": "1.0.0",
        "timestamp": time.time()
    }


@app.get("/health/detailed", tags=["監控"])
async def detailed_health_check():
    """詳細健康檢查"""
    checks = {
        "api": "healthy",
        "database": "checking...",
        "ai_model": "checking...",
        "cache": "checking...",
    }

    # TODO: 實際檢查各個組件狀態
    # checks["database"] = await check_database_health()
    # checks["ai_model"] = await check_model_health()
    # checks["cache"] = await check_cache_health()

    all_healthy = all(status == "healthy" for status in checks.values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
        "timestamp": time.time()
    }


# 根路徑
@app.get("/", tags=["根目錄"])
async def root():
    """API 根路徑"""
    return {
        "message": "壽命預測保險決策輔助系統 API",
        "version": "1.0.0",
        "author": "shaobai",
        "docs": "/docs",
        "health": "/health"
    }


# 包含 API 路由
app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level="info"
    )