"""
圖像上傳 API 路由

處理面相圖像上傳、驗證和預處理
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, List
import asyncio
import io
import logging
import time
from pathlib import Path
import hashlib
import shutil
from PIL import Image
import numpy as np

from ..core.database import get_db
from ..core.security import get_current_user
from ..core.config import get_settings
from ..models.user import User
from ..schemas.upload import ImageUploadResponse, PredictionRequest, PredictionResponse
from ..services.image_service import ImageService
from ..services.prediction_service import PredictionService
from ..middleware.rate_limit import RateLimitConfig, rate_limit

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/upload", tags=["圖像上傳"])

# 速率限制配置
upload_rate_limit = RateLimitConfig(max_requests=10, time_window=3600)  # 1小時10次
predict_rate_limit = RateLimitConfig(max_requests=3, time_window=3600)  # 1小時3次預測


@router.post("/image", response_model=ImageUploadResponse)
@rate_limit(upload_rate_limit)
async def upload_image(
    file: UploadFile = File(..., description="面相圖像文件"),
    description: Optional[str] = Form(None, description="圖像描述"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    上傳面相圖像

    - **file**: 支持 JPG, JPEG, PNG 格式
    - **description**: 可選的圖像描述
    - **限制**: 最大5MB，每小時10次
    """
    try:
        image_service = ImageService()

        # 驗證用戶狀態
        if not current_user.is_verified:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="請先驗證電子郵件才能使用預測功能"
            )

        # 驗證文件
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="請提供有效的文件"
            )

        # 檢查文件擴展名
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的文件格式。支持格式：{', '.join(settings.ALLOWED_EXTENSIONS)}"
            )

        # 讀取文件內容
        file_content = await file.read()

        # 檢查文件大小
        if len(file_content) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"文件過大。最大允許 {settings.MAX_UPLOAD_SIZE // 1024 // 1024}MB"
            )

        # 驗證圖像內容
        try:
            image = Image.open(io.BytesIO(file_content))
            image.verify()  # 驗證圖像完整性

            # 重新打開用於處理（verify會關閉文件）
            image = Image.open(io.BytesIO(file_content))

            # 轉換為RGB（如果是RGBA或其他格式）
            if image.mode != 'RGB':
                image = image.convert('RGB')

        except Exception as e:
            logger.error(f"圖像驗證失敗: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="無效的圖像文件"
            )

        # 基本圖像質量檢查
        width, height = image.size
        if width < 100 or height < 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="圖像尺寸過小，至少需要100x100像素"
            )

        if width > 4000 or height > 4000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="圖像尺寸過大，最大支持4000x4000像素"
            )

        # 處理圖像
        processed_result = await image_service.process_uploaded_image(
            image=image,
            user_id=current_user.user_id,
            description=description
        )

        if not processed_result['success']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=processed_result.get('message', '圖像處理失敗')
            )

        logger.info(f"用戶 {current_user.email} 成功上傳圖像")

        return ImageUploadResponse(
            success=True,
            message="圖像上傳成功",
            image_id=processed_result['image_id'],
            image_info=processed_result['image_info'],
            face_detected=processed_result.get('face_detected', False),
            processing_time=processed_result.get('processing_time', 0)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"圖像上傳失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="圖像上傳處理失敗，請稍後再試"
        )


@router.post("/predict", response_model=PredictionResponse)
@rate_limit(predict_rate_limit)
async def predict_lifespan(
    request: PredictionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    基於上傳的圖像進行壽命預測

    - **image_id**: 通過上傳接口獲得的圖像ID
    - **限制**: 每小時3次預測
    - **要求**: 已驗證的用戶且有剩餘預測次數
    """
    try:
        # 檢查用戶狀態
        if not current_user.is_verified:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="請先驗證電子郵件"
            )

        # 檢查預測額度
        if not current_user.can_predict():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="預測次數已用完。請升級為高級會員獲得無限預測次數"
            )

        # 初始化預測服務
        prediction_service = PredictionService()

        # 執行預測
        prediction_result = await prediction_service.predict_from_image_id(
            image_id=request.image_id,
            user_id=current_user.user_id,
            options=request.options
        )

        if not prediction_result['success']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=prediction_result.get('message', '預測失敗')
            )

        # 使用預測額度
        from ..services.user_service import UserService
        user_service = UserService(db)
        if not user_service.use_prediction_quota(current_user.user_id):
            logger.warning(f"預測額度使用失敗: {current_user.user_id}")

        logger.info(f"用戶 {current_user.email} 完成壽命預測")

        return PredictionResponse(
            success=True,
            message="預測完成",
            prediction_id=prediction_result['prediction_id'],
            results=prediction_result['results'],
            confidence=prediction_result['confidence'],
            processing_time=prediction_result.get('processing_time', 0),
            remaining_predictions=current_user.remaining_predictions - 1 if not current_user.is_premium else -1
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"壽命預測失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="預測處理失敗，請稍後再試"
        )


@router.get("/history", response_model=List[Dict[str, Any]])
async def get_upload_history(
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    獲取用戶的上傳和預測歷史

    - **limit**: 返回記錄數量限制 (預設10, 最大100)
    - **offset**: 分頁偏移量
    """
    try:
        if limit > 100:
            limit = 100

        image_service = ImageService()
        history = await image_service.get_user_history(
            user_id=current_user.user_id,
            limit=limit,
            offset=offset
        )

        return history

    except Exception as e:
        logger.error(f"獲取歷史記錄失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="獲取歷史記錄失敗"
        )


@router.delete("/image/{image_id}")
async def delete_image(
    image_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    刪除上傳的圖像

    - **image_id**: 要刪除的圖像ID
    - 只能刪除自己上傳的圖像
    """
    try:
        image_service = ImageService()

        result = await image_service.delete_user_image(
            image_id=image_id,
            user_id=current_user.user_id
        )

        if not result['success']:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.get('message', '圖像不存在或無權限刪除')
            )

        logger.info(f"用戶 {current_user.email} 刪除圖像: {image_id}")

        return {
            "success": True,
            "message": "圖像已刪除"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"刪除圖像失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="刪除失敗"
        )


@router.get("/limits")
async def get_upload_limits(current_user: User = Depends(get_current_user)):
    """
    獲取當前用戶的上傳限制和使用情況
    """
    try:
        return {
            "max_file_size_mb": settings.MAX_UPLOAD_SIZE // 1024 // 1024,
            "allowed_extensions": settings.ALLOWED_EXTENSIONS,
            "hourly_upload_limit": 10,
            "hourly_prediction_limit": 3,
            "remaining_predictions": current_user.remaining_predictions if not current_user.is_premium else -1,
            "is_premium": current_user.is_premium,
            "is_verified": current_user.is_verified
        }

    except Exception as e:
        logger.error(f"獲取上傳限制失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="獲取限制信息失敗"
        )