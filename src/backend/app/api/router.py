"""
API 路由器

統一管理所有 API 端點路由
"""

from fastapi import APIRouter

# 導入各個模組的路由器
# from app.api.endpoints import auth, users, predictions, health

# 建立主要 API 路由器
api_router = APIRouter()

# 註冊各個端點路由
# api_router.include_router(auth.router, prefix="/auth", tags=["認證"])
# api_router.include_router(users.router, prefix="/users", tags=["用戶管理"])
# api_router.include_router(predictions.router, prefix="/predict", tags=["預測"])
# api_router.include_router(health.router, prefix="/health", tags=["健康檢查"])

# 暫時的測試端點
@api_router.get("/", tags=["測試"])
async def api_root():
    """API 根端點"""
    return {
        "message": "壽命預測 API v1.0",
        "author": "shaobai",
        "endpoints": {
            "auth": "/api/v1/auth",
            "users": "/api/v1/users",
            "predict": "/api/v1/predict",
            "health": "/api/v1/health"
        },
        "status": "開發中"
    }


@api_router.get("/status", tags=["測試"])
async def api_status():
    """API 狀態檢查"""
    return {
        "api_version": "1.0.0",
        "status": "運行中",
        "features": {
            "authentication": "規劃中",
            "face_upload": "規劃中",
            "life_prediction": "規劃中",
            "history": "規劃中"
        }
    }