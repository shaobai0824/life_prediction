"""
預測服務層

處理壽命預測的業務邏輯和AI模型調用
"""

import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import json

import numpy as np

from ..core.config import get_settings
from ..ml_pipeline.inference.prediction_pipeline import PredictionPipeline
from .image_service import ImageService

logger = logging.getLogger(__name__)
settings = get_settings()


class PredictionService:
    """預測服務類"""

    def __init__(self):
        self.settings = settings
        self.image_service = ImageService()

        # 初始化預測管道
        try:
            self.prediction_pipeline = PredictionPipeline(
                device=settings.DEVICE,
                models_dir=settings.MODEL_PATH
            )
            logger.info("預測管道初始化成功")
        except Exception as e:
            logger.error(f"預測管道初始化失敗: {e}")
            self.prediction_pipeline = None

        # 預測記錄緩存（簡化版本）
        self._prediction_cache = {}
        self._user_predictions = {}  # user_id -> [prediction_records]

    async def predict_from_image_id(
        self,
        image_id: str,
        user_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """從圖像ID進行預測"""
        try:
            start_time = datetime.utcnow()

            # 獲取圖像數據
            image_data = await self.image_service.get_image_data(image_id, user_id)
            if not image_data:
                return {
                    'success': False,
                    'message': '圖像不存在或無權限訪問'
                }

            # 檢查是否檢測到面部
            if not image_data['record']['face_detected']:
                return {
                    'success': False,
                    'message': '未在圖像中檢測到清晰的面部，請上傳正面清晰的人臉照片'
                }

            # 檢查面部質量
            face_quality = image_data['record'].get('face_quality', 0)
            if face_quality < 0.5:
                return {
                    'success': False,
                    'message': '面部圖像質量不足，請上傳更清晰的照片'
                }

            # 生成預測ID
            prediction_id = str(uuid.uuid4())

            # 設定預測選項
            prediction_options = self._prepare_prediction_options(options)

            # 執行預測
            if self.prediction_pipeline is None:
                # 模型未訓練，使用模擬預測
                prediction_result = await self._mock_prediction(
                    image_data, prediction_options
                )
            else:
                # 使用真實AI模型預測
                prediction_result = await self._ai_prediction(
                    image_data, prediction_options
                )

            if not prediction_result['success']:
                return prediction_result

            # 計算處理時間
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            # 創建預測記錄
            prediction_record = {
                'prediction_id': prediction_id,
                'user_id': user_id,
                'image_id': image_id,
                'created_at': start_time,
                'completed_at': datetime.utcnow(),
                'processing_time': processing_time,
                'options': prediction_options,
                'results': prediction_result['results'],
                'confidence': prediction_result['confidence'],
                'status': 'completed',
                'model_version': prediction_result.get('model_version', 'mock-1.0'),
                'metadata': {
                    'face_quality': face_quality,
                    'image_quality': image_data['record'].get('processing_metadata', {}).get('quality_score', 0),
                    'processing_notes': prediction_result.get('notes', [])
                }
            }

            # 保存預測記錄
            self._prediction_cache[prediction_id] = prediction_record

            # 更新用戶預測歷史
            if user_id not in self._user_predictions:
                self._user_predictions[user_id] = []
            self._user_predictions[user_id].append(prediction_record)

            logger.info(f"預測完成: {prediction_id}, 用戶: {user_id}, 壽命: {prediction_result['results']['life_prediction']['predicted_lifespan']:.1f}歲")

            return {
                'success': True,
                'message': '預測完成',
                'prediction_id': prediction_id,
                'results': prediction_result['results'],
                'confidence': prediction_result['confidence'],
                'processing_time': processing_time
            }

        except Exception as e:
            logger.error(f"預測執行失敗: {e}")
            return {
                'success': False,
                'message': f'預測過程中發生錯誤: {str(e)}'
            }

    def _prepare_prediction_options(self, options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """準備預測選項"""
        default_options = {
            'include_health_analysis': True,
            'include_lifestyle_factors': True,
            'include_genetic_factors': False,
            'detailed_report': True,
            'confidence_threshold': 0.6
        }

        if options:
            default_options.update(options)

        return default_options

    async def _ai_prediction(
        self,
        image_data: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用AI模型進行預測"""
        try:
            processed_image = image_data['processed_image']

            # 使用預測管道
            pipeline_result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.prediction_pipeline.predict_from_image,
                processed_image
            )

            if not pipeline_result['success']:
                return {
                    'success': False,
                    'message': pipeline_result.get('message', 'AI預測失敗')
                }

            # 解析預測結果
            result_data = pipeline_result['result']
            metadata = pipeline_result.get('metadata', {})

            # 構建回應格式
            results = {
                'life_prediction': {
                    'predicted_lifespan': result_data['life_prediction']['predicted_lifespan'],
                    'confidence': result_data['life_prediction']['confidence'],
                    'prediction_range': result_data['life_prediction']['prediction_range'],
                    'uncertainty': result_data['life_prediction']['uncertainty']
                },
                'face_analysis': {
                    'faces_detected': result_data['face_analysis']['faces_detected'],
                    'primary_face_confidence': result_data['face_analysis']['primary_face_confidence'],
                    'face_characteristics': result_data['face_analysis']['face_characteristics']
                },
                'confidence_assessment': result_data['confidence_assessment']
            }

            # 添加詳細分析（如果需要）
            if options.get('include_health_analysis', True):
                results['detailed_analysis'] = result_data.get('detailed_analysis', {})

            return {
                'success': True,
                'results': results,
                'confidence': result_data['confidence_assessment']['overall_confidence'],
                'model_version': metadata.get('model_versions', {}),
                'notes': ['使用AI模型預測']
            }

        except Exception as e:
            logger.error(f"AI預測失敗: {e}")
            return {
                'success': False,
                'message': f'AI預測過程中發生錯誤: {str(e)}'
            }

    async def _mock_prediction(
        self,
        image_data: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """模擬預測（模型未訓練時使用）"""
        try:
            # 模擬處理延遲
            await asyncio.sleep(1.0)

            # 基於圖像特徵進行簡單估算
            face_quality = image_data['record'].get('face_quality', 0.5)
            image_quality = image_data['record'].get('processing_metadata', {}).get('quality_score', 0.5)

            # 基礎壽命估算（使用統計平均值 + 隨機變化）
            base_lifespan = 78.0  # 全球平均壽命
            quality_bonus = (face_quality + image_quality) * 5  # 高質量圖像加分
            random_variation = np.random.normal(0, 8)  # 隨機變異

            predicted_lifespan = max(50, min(100, base_lifespan + quality_bonus + random_variation))
            uncertainty = 10.0 + (1.0 - min(face_quality, image_quality)) * 10.0

            # 計算信心度
            overall_confidence = min(face_quality * image_quality * 1.2, 0.85)

            # 構建模擬結果
            results = {
                'life_prediction': {
                    'predicted_lifespan': round(predicted_lifespan, 1),
                    'confidence': round(overall_confidence, 3),
                    'prediction_range': {
                        'min': round(predicted_lifespan - uncertainty, 1),
                        'max': round(predicted_lifespan + uncertainty, 1)
                    },
                    'uncertainty': round(uncertainty, 1)
                },
                'face_analysis': {
                    'faces_detected': image_data['record']['face_count'],
                    'primary_face_confidence': face_quality,
                    'face_characteristics': {
                        'age_indicator': round(np.random.uniform(0.4, 0.8), 3),
                        'health_indicator': round(face_quality * 0.8 + 0.1, 3),
                        'facial_structure_score': round(np.random.uniform(0.5, 0.9), 3),
                        'skin_quality_score': round(image_quality * 0.7 + 0.2, 3),
                        'expression_score': round(np.random.uniform(0.3, 0.7), 3)
                    }
                },
                'confidence_assessment': {
                    'overall_confidence': round(overall_confidence, 3),
                    'confidence_level': self._get_confidence_level(overall_confidence),
                    'reliability_assessment': self._get_reliability_assessment(overall_confidence),
                    'component_confidences': {
                        'face_detection': round(face_quality, 3),
                        'feature_extraction': round(image_quality, 3),
                        'life_prediction': round(overall_confidence * 0.8, 3)
                    }
                }
            }

            # 添加健康分析（如果需要）
            if options.get('include_health_analysis', True):
                results['detailed_analysis'] = {
                    'health_risk': {
                        'level': np.random.choice(['低', '中等', '中等', '中等', '高'], p=[0.2, 0.3, 0.3, 0.15, 0.05]),
                        'score': round(np.random.uniform(0.4, 0.8), 3),
                        'distribution': [round(x, 3) for x in np.random.dirichlet([2, 3, 4, 2, 1])]
                    },
                    'lifestyle_factors': {
                        '運動習慣': {
                            'score': round(np.random.uniform(0.3, 0.9), 3),
                            'level': np.random.choice(['較差', '中等', '良好', '優秀'])
                        },
                        '飲食質量': {
                            'score': round(np.random.uniform(0.4, 0.8), 3),
                            'level': np.random.choice(['中等', '良好', '優秀'])
                        },
                        '壓力管理': {
                            'score': round(np.random.uniform(0.3, 0.7), 3),
                            'level': np.random.choice(['較差', '中等', '良好'])
                        },
                        '睡眠質量': {
                            'score': round(np.random.uniform(0.4, 0.9), 3),
                            'level': np.random.choice(['中等', '良好', '優秀'])
                        }
                    }
                }

                if options.get('include_genetic_factors', False):
                    results['detailed_analysis']['genetic_factors'] = {
                        '長壽基因': {
                            'score': round(np.random.uniform(0.4, 0.8), 3),
                            'level': np.random.choice(['中等', '良好'])
                        },
                        '疾病易感性': {
                            'score': round(np.random.uniform(0.3, 0.7), 3),
                            'level': np.random.choice(['較低', '中等'])
                        },
                        '代謝效率': {
                            'score': round(np.random.uniform(0.5, 0.9), 3),
                            'level': np.random.choice(['良好', '優秀'])
                        }
                    }

            return {
                'success': True,
                'results': results,
                'confidence': overall_confidence,
                'model_version': 'mock-simulation-1.0',
                'notes': [
                    '此為模擬預測結果，僅供展示',
                    'AI模型正在訓練中，實際結果可能不同',
                    '請勿將此結果用於實際決策'
                ]
            }

        except Exception as e:
            logger.error(f"模擬預測失敗: {e}")
            return {
                'success': False,
                'message': f'模擬預測失敗: {str(e)}'
            }

    def _get_confidence_level(self, confidence: float) -> str:
        """獲取信心度等級"""
        if confidence >= 0.8:
            return "高"
        elif confidence >= 0.6:
            return "中等"
        else:
            return "低"

    def _get_reliability_assessment(self, confidence: float) -> str:
        """獲取可靠性評估"""
        if confidence >= 0.8:
            return "預測結果可信度較高"
        elif confidence >= 0.6:
            return "預測結果具有一定參考價值"
        else:
            return "預測結果僅供初步參考"

    async def get_prediction_history(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """獲取用戶預測歷史"""
        try:
            if user_id not in self._user_predictions:
                return []

            # 按創建時間排序
            user_predictions = sorted(
                self._user_predictions[user_id],
                key=lambda x: x['created_at'],
                reverse=True
            )

            # 分頁
            start = offset
            end = offset + limit
            page_predictions = user_predictions[start:end]

            # 格式化返回數據
            history = []
            for pred in page_predictions:
                history.append({
                    'prediction_id': pred['prediction_id'],
                    'image_id': pred['image_id'],
                    'created_at': pred['created_at'],
                    'predicted_lifespan': pred['results']['life_prediction']['predicted_lifespan'],
                    'confidence': pred['confidence'],
                    'status': pred['status'],
                    'processing_time': pred['processing_time']
                })

            return history

        except Exception as e:
            logger.error(f"獲取預測歷史失敗: {e}")
            return []

    async def get_prediction_detail(
        self,
        prediction_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """獲取預測詳情"""
        try:
            if prediction_id not in self._prediction_cache:
                return None

            prediction = self._prediction_cache[prediction_id]

            # 驗證權限
            if prediction['user_id'] != user_id:
                logger.warning(f"用戶 {user_id} 嘗試存取不屬於自己的預測 {prediction_id}")
                return None

            return prediction

        except Exception as e:
            logger.error(f"獲取預測詳情失敗: {e}")
            return None

    def get_service_stats(self) -> Dict[str, Any]:
        """獲取服務統計資訊"""
        return {
            'total_predictions': len(self._prediction_cache),
            'total_users': len(self._user_predictions),
            'pipeline_available': self.prediction_pipeline is not None,
            'model_path': str(self.settings.MODEL_PATH),
            'device': self.settings.DEVICE
        }