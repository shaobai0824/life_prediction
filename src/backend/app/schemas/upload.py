"""
上傳相關的 Pydantic 模型

定義圖像上傳和預測的請求回應格式
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime


class ImageUploadResponse(BaseModel):
    """圖像上傳回應"""
    success: bool
    message: str
    image_id: str = Field(..., description="圖像唯一識別碼")
    image_info: Dict[str, Any] = Field(..., description="圖像資訊")
    face_detected: bool = Field(..., description="是否檢測到面部")
    processing_time: float = Field(..., description="處理時間（秒）")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "圖像上傳成功",
                "image_id": "uuid-1234-5678-9012",
                "image_info": {
                    "width": 800,
                    "height": 600,
                    "format": "JPEG",
                    "file_size": 245760
                },
                "face_detected": True,
                "processing_time": 2.5
            }
        }


class PredictionOptions(BaseModel):
    """預測選項"""
    include_health_analysis: Optional[bool] = Field(True, description="包含健康分析")
    include_lifestyle_factors: Optional[bool] = Field(True, description="包含生活方式分析")
    include_genetic_factors: Optional[bool] = Field(False, description="包含遺傳因子分析")
    detailed_report: Optional[bool] = Field(True, description="生成詳細報告")
    confidence_threshold: Optional[float] = Field(0.6, ge=0.0, le=1.0, description="信心度閾值")

    class Config:
        schema_extra = {
            "example": {
                "include_health_analysis": True,
                "include_lifestyle_factors": True,
                "include_genetic_factors": False,
                "detailed_report": True,
                "confidence_threshold": 0.7
            }
        }


class PredictionRequest(BaseModel):
    """預測請求"""
    image_id: str = Field(..., description="圖像ID")
    options: Optional[PredictionOptions] = Field(None, description="預測選項")

    class Config:
        schema_extra = {
            "example": {
                "image_id": "uuid-1234-5678-9012",
                "options": {
                    "include_health_analysis": True,
                    "detailed_report": True
                }
            }
        }


class LifePredictionResult(BaseModel):
    """壽命預測結果"""
    predicted_lifespan: float = Field(..., description="預測壽命（歲）")
    confidence: float = Field(..., ge=0.0, le=1.0, description="預測信心度")
    prediction_range: Dict[str, float] = Field(..., description="預測範圍")
    uncertainty: float = Field(..., description="預測不確定性")

    class Config:
        schema_extra = {
            "example": {
                "predicted_lifespan": 82.5,
                "confidence": 0.78,
                "prediction_range": {
                    "min": 75.2,
                    "max": 89.8
                },
                "uncertainty": 7.3
            }
        }


class FaceAnalysis(BaseModel):
    """面部分析結果"""
    faces_detected: int = Field(..., description="檢測到的面部數量")
    primary_face_confidence: float = Field(..., description="主要面部信心度")
    face_characteristics: Dict[str, Any] = Field(..., description="面部特徵")

    class Config:
        schema_extra = {
            "example": {
                "faces_detected": 1,
                "primary_face_confidence": 0.95,
                "face_characteristics": {
                    "age_indicator": 0.65,
                    "health_indicator": 0.72,
                    "facial_structure_score": 0.68
                }
            }
        }


class HealthAnalysis(BaseModel):
    """健康分析結果"""
    health_risk: Dict[str, Any] = Field(..., description="健康風險評估")
    lifestyle_factors: Dict[str, Any] = Field(..., description="生活方式因子")
    genetic_factors: Optional[Dict[str, Any]] = Field(None, description="遺傳因子")

    class Config:
        schema_extra = {
            "example": {
                "health_risk": {
                    "level": "中等",
                    "score": 0.65,
                    "distribution": [0.1, 0.2, 0.4, 0.25, 0.05]
                },
                "lifestyle_factors": {
                    "運動習慣": {"score": 0.7, "level": "良好"},
                    "飲食質量": {"score": 0.6, "level": "中等"},
                    "壓力管理": {"score": 0.5, "level": "中等"},
                    "睡眠質量": {"score": 0.8, "level": "優秀"}
                }
            }
        }


class ConfidenceAssessment(BaseModel):
    """信心度評估"""
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="整體信心度")
    confidence_level: str = Field(..., description="信心度等級")
    reliability_assessment: str = Field(..., description="可靠性評估")
    component_confidences: Dict[str, float] = Field(..., description="各組件信心度")

    class Config:
        schema_extra = {
            "example": {
                "overall_confidence": 0.75,
                "confidence_level": "高",
                "reliability_assessment": "預測結果可信度較高",
                "component_confidences": {
                    "face_detection": 0.95,
                    "feature_extraction": 0.82,
                    "life_prediction": 0.68
                }
            }
        }


class PredictionResults(BaseModel):
    """完整預測結果"""
    life_prediction: LifePredictionResult
    face_analysis: FaceAnalysis
    detailed_analysis: Optional[HealthAnalysis] = None
    confidence_assessment: ConfidenceAssessment

    class Config:
        schema_extra = {
            "example": {
                "life_prediction": {
                    "predicted_lifespan": 82.5,
                    "confidence": 0.78,
                    "prediction_range": {"min": 75.2, "max": 89.8},
                    "uncertainty": 7.3
                },
                "face_analysis": {
                    "faces_detected": 1,
                    "primary_face_confidence": 0.95,
                    "face_characteristics": {
                        "age_indicator": 0.65,
                        "health_indicator": 0.72
                    }
                },
                "confidence_assessment": {
                    "overall_confidence": 0.75,
                    "confidence_level": "高",
                    "reliability_assessment": "預測結果可信度較高",
                    "component_confidences": {
                        "face_detection": 0.95,
                        "feature_extraction": 0.82,
                        "life_prediction": 0.68
                    }
                }
            }
        }


class PredictionResponse(BaseModel):
    """預測回應"""
    success: bool
    message: str
    prediction_id: str = Field(..., description="預測記錄ID")
    results: PredictionResults
    confidence: float = Field(..., ge=0.0, le=1.0, description="整體信心度")
    processing_time: float = Field(..., description="處理時間（秒）")
    remaining_predictions: int = Field(..., description="剩餘預測次數，-1表示無限制")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "預測完成",
                "prediction_id": "pred-1234-5678-9012",
                "results": {
                    "life_prediction": {
                        "predicted_lifespan": 82.5,
                        "confidence": 0.78
                    }
                },
                "confidence": 0.75,
                "processing_time": 5.2,
                "remaining_predictions": 2
            }
        }


class ImageInfo(BaseModel):
    """圖像資訊"""
    image_id: str
    filename: str
    upload_time: datetime
    file_size: int
    width: int
    height: int
    format: str
    face_detected: bool
    has_prediction: bool
    description: Optional[str] = None

    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "image_id": "uuid-1234-5678-9012",
                "filename": "portrait.jpg",
                "upload_time": "2024-01-25T10:30:00Z",
                "file_size": 245760,
                "width": 800,
                "height": 600,
                "format": "JPEG",
                "face_detected": True,
                "has_prediction": True,
                "description": "正面清晰照片"
            }
        }


class PredictionHistory(BaseModel):
    """預測歷史記錄"""
    prediction_id: str
    image_id: str
    created_at: datetime
    predicted_lifespan: float
    confidence: float
    status: str  # completed, processing, failed

    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "prediction_id": "pred-1234-5678-9012",
                "image_id": "img-1234-5678-9012",
                "created_at": "2024-01-25T10:30:00Z",
                "predicted_lifespan": 82.5,
                "confidence": 0.78,
                "status": "completed"
            }
        }


class UserHistory(BaseModel):
    """用戶歷史記錄"""
    images: List[ImageInfo]
    predictions: List[PredictionHistory]
    total_uploads: int
    total_predictions: int

    class Config:
        schema_extra = {
            "example": {
                "images": [
                    {
                        "image_id": "img-1234",
                        "filename": "photo1.jpg",
                        "upload_time": "2024-01-25T10:30:00Z",
                        "face_detected": True,
                        "has_prediction": True
                    }
                ],
                "predictions": [
                    {
                        "prediction_id": "pred-1234",
                        "image_id": "img-1234",
                        "created_at": "2024-01-25T10:35:00Z",
                        "predicted_lifespan": 82.5,
                        "confidence": 0.78,
                        "status": "completed"
                    }
                ],
                "total_uploads": 5,
                "total_predictions": 3
            }
        }