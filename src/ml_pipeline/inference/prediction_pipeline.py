"""
完整的預測管道

整合面部檢測、特徵提取和壽命預測的完整流程
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Union, Optional
from pathlib import Path
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..models.face_detection.detector import FaceDetector
from ..models.feature_extraction.extractor import FaceFeatureExtractor
from ..models.life_prediction.predictor import LifePredictionModel
from ..data_processing.image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """完整的預測管道"""

    def __init__(self, device: str = None, models_dir: str = "./models/trained"):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models_dir = Path(models_dir)

        # 初始化各個組件
        self.face_detector = None
        self.feature_extractor = None
        self.life_predictor = None
        self.image_processor = None

        # 管道配置
        self.config = {
            'face_detection': {
                'min_face_size': 40,
                'confidence_threshold': 0.7
            },
            'feature_extraction': {
                'feature_dim': 512,
                'target_size': (224, 224)
            },
            'life_prediction': {
                'confidence_threshold': 0.6,
                'min_age': 0,
                'max_age': 120
            }
        }

        # 載入模型
        self._initialize_models()

    def _initialize_models(self):
        """初始化所有模型"""
        try:
            logger.info("初始化預測管道...")

            # 初始化面部檢測器
            self.face_detector = FaceDetector(
                min_face_size=self.config['face_detection']['min_face_size'],
                device=str(self.device)
            )

            # 初始化特徵提取器
            self.feature_extractor = FaceFeatureExtractor(
                feature_dim=self.config['feature_extraction']['feature_dim'],
                device=str(self.device)
            )

            # 初始化壽命預測器
            self.life_predictor = LifePredictionModel(
                input_dim=self.config['feature_extraction']['feature_dim'],
                device=str(self.device)
            )

            # 初始化圖像處理器
            self.image_processor = ImageProcessor()

            # 嘗試載入預訓練模型
            self._load_pretrained_models()

            logger.info("預測管道初始化成功")

        except Exception as e:
            logger.error(f"模型初始化失敗: {e}")
            raise

    def _load_pretrained_models(self):
        """載入預訓練模型"""
        try:
            # 載入面部檢測器（使用 OpenCV Haar Cascade，無需載入）
            logger.info("面部檢測器準備就緒")

            # 載入特徵提取器
            feature_model_path = self.models_dir / "feature_extractor.pth"
            if feature_model_path.exists():
                if self.feature_extractor.load_model(feature_model_path):
                    logger.info("特徵提取器載入成功")
                else:
                    logger.warning("特徵提取器載入失敗，使用未訓練模型")
            else:
                logger.warning("特徵提取器模型檔案不存在，使用未訓練模型")

            # 載入壽命預測器
            predictor_model_path = self.models_dir / "life_predictor.pth"
            if predictor_model_path.exists():
                if self.life_predictor.load_model(predictor_model_path):
                    logger.info("壽命預測器載入成功")
                else:
                    logger.warning("壽命預測器載入失敗，使用未訓練模型")
            else:
                logger.warning("壽命預測器模型檔案不存在，使用未訓練模型")

        except Exception as e:
            logger.warning(f"預訓練模型載入警告: {e}")

    async def predict_from_image_async(self, image_input: Union[np.ndarray, str, Path]) -> Dict[str, Any]:
        """非同步預測"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, self.predict_from_image, image_input)
            return result

    def predict_from_image(self, image_input: Union[np.ndarray, str, Path]) -> Dict[str, Any]:
        """從圖像進行完整的壽命預測"""
        start_time = time.time()

        try:
            logger.info("開始壽命預測流程...")

            # 步驟 1: 圖像預處理和驗證
            processed_image = self._preprocess_image(image_input)
            if processed_image is None:
                return self._create_error_response("圖像處理失敗")

            # 步驟 2: 面部檢測
            face_result = self.face_detector.predict(processed_image)
            if not face_result['success'] or not face_result['faces']:
                return self._create_error_response(
                    face_result.get('message', '未檢測到面部'),
                    step='face_detection'
                )

            # 步驟 3: 提取主要面部區域
            primary_face = face_result['primary_face']
            face_regions = face_result.get('face_regions', [])

            if not face_regions:
                return self._create_error_response("無法提取面部區域")

            main_face_region = face_regions[0]  # 使用最大的面部

            # 步驟 4: 特徵提取
            feature_result = self.feature_extractor.predict(main_face_region)
            if not feature_result['success']:
                return self._create_error_response(
                    feature_result.get('message', '特徵提取失敗'),
                    step='feature_extraction'
                )

            # 步驟 5: 壽命預測
            prediction_result = self.life_predictor.predict(feature_result['features'])
            if not prediction_result['success']:
                return self._create_error_response(
                    prediction_result.get('message', '壽命預測失敗'),
                    step='life_prediction'
                )

            # 步驟 6: 整合結果
            integrated_result = self._integrate_results(
                face_result, feature_result, prediction_result
            )

            # 計算總處理時間
            processing_time = time.time() - start_time

            return {
                'success': True,
                'message': '預測完成',
                'processing_time': round(processing_time, 3),
                'result': integrated_result,
                'metadata': {
                    'model_versions': {
                        'face_detector': self.face_detector.version,
                        'feature_extractor': self.feature_extractor.version,
                        'life_predictor': self.life_predictor.version
                    },
                    'device': str(self.device),
                    'timestamp': time.time()
                }
            }

        except Exception as e:
            logger.error(f"預測流程失敗: {e}")
            return self._create_error_response(f"預測過程中發生錯誤: {str(e)}")

    def _preprocess_image(self, image_input: Union[np.ndarray, str, Path]) -> Optional[np.ndarray]:
        """預處理輸入圖像"""
        try:
            if isinstance(image_input, (str, Path)):
                # 從檔案載入圖像
                image_path = Path(image_input)
                if not image_path.exists():
                    logger.error(f"圖像檔案不存在: {image_path}")
                    return None

                image = cv2.imread(str(image_path))
                if image is None:
                    logger.error("無法讀取圖像檔案")
                    return None

                # 轉換為 RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            elif isinstance(image_input, np.ndarray):
                image = image_input.copy()
            else:
                logger.error("不支援的圖像輸入類型")
                return None

            # 使用圖像處理器進行預處理
            if self.image_processor:
                image = self.image_processor.enhance_image(image)

            # 驗證圖像
            if not self._validate_image(image):
                return None

            return image

        except Exception as e:
            logger.error(f"圖像預處理失敗: {e}")
            return None

    def _validate_image(self, image: np.ndarray) -> bool:
        """驗證圖像是否適合處理"""
        try:
            # 檢查圖像維度
            if len(image.shape) != 3 or image.shape[2] != 3:
                logger.error("圖像必須是3通道RGB格式")
                return False

            # 檢查圖像尺寸
            height, width = image.shape[:2]
            if height < 100 or width < 100:
                logger.error("圖像尺寸太小")
                return False

            if height > 4000 or width > 4000:
                logger.error("圖像尺寸太大")
                return False

            # 檢查圖像內容
            if np.all(image == 0) or np.all(image == 255):
                logger.error("圖像內容無效")
                return False

            return True

        except Exception as e:
            logger.error(f"圖像驗證失敗: {e}")
            return False

    def _integrate_results(self, face_result: Dict, feature_result: Dict,
                          prediction_result: Dict) -> Dict[str, Any]:
        """整合所有預測結果"""
        try:
            prediction_data = prediction_result['prediction']

            return {
                'life_prediction': {
                    'predicted_lifespan': prediction_data['predicted_lifespan'],
                    'confidence': prediction_data['confidence'],
                    'prediction_range': prediction_data['prediction_range'],
                    'uncertainty': prediction_data['uncertainty']
                },
                'face_analysis': {
                    'faces_detected': len(face_result['faces']),
                    'primary_face_confidence': face_result['primary_face']['confidence'],
                    'face_characteristics': feature_result.get('characteristics', {})
                },
                'detailed_analysis': prediction_data.get('analyses', {}),
                'health_report': prediction_result.get('report', {}),
                'confidence_assessment': self._assess_overall_confidence(
                    face_result, feature_result, prediction_result
                )
            }

        except Exception as e:
            logger.error(f"結果整合失敗: {e}")
            return {}

    def _assess_overall_confidence(self, face_result: Dict, feature_result: Dict,
                                 prediction_result: Dict) -> Dict[str, Any]:
        """評估整體預測信心度"""
        try:
            # 各階段信心度
            face_confidence = face_result['primary_face']['confidence']
            feature_confidence = feature_result.get('confidence', 0.0)
            prediction_confidence = prediction_result['prediction']['confidence']

            # 計算整體信心度（加權平均）
            overall_confidence = (
                face_confidence * 0.2 +
                feature_confidence * 0.3 +
                prediction_confidence * 0.5
            )

            # 信心度等級
            if overall_confidence >= 0.8:
                confidence_level = "高"
                reliability = "預測結果可信度較高"
            elif overall_confidence >= 0.6:
                confidence_level = "中等"
                reliability = "預測結果具有一定參考價值"
            else:
                confidence_level = "低"
                reliability = "預測結果僅供初步參考"

            return {
                'overall_confidence': round(overall_confidence, 3),
                'confidence_level': confidence_level,
                'reliability_assessment': reliability,
                'component_confidences': {
                    'face_detection': round(face_confidence, 3),
                    'feature_extraction': round(feature_confidence, 3),
                    'life_prediction': round(prediction_confidence, 3)
                }
            }

        except Exception as e:
            logger.error(f"信心度評估失敗: {e}")
            return {'overall_confidence': 0.0, 'confidence_level': "未知"}

    def _create_error_response(self, message: str, step: str = None) -> Dict[str, Any]:
        """建立錯誤回應"""
        return {
            'success': False,
            'message': message,
            'failed_step': step,
            'result': None,
            'timestamp': time.time()
        }

    def get_pipeline_info(self) -> Dict[str, Any]:
        """獲取管道資訊"""
        return {
            'device': str(self.device),
            'models': {
                'face_detector': self.face_detector.get_model_info() if self.face_detector else None,
                'feature_extractor': self.feature_extractor.get_model_info() if self.feature_extractor else None,
                'life_predictor': self.life_predictor.get_model_info() if self.life_predictor else None
            },
            'config': self.config
        }

    def update_config(self, new_config: Dict[str, Any]):
        """更新管道配置"""
        self.config.update(new_config)
        logger.info("管道配置已更新")


def create_prediction_pipeline(device: str = None, models_dir: str = "./models/trained") -> PredictionPipeline:
    """建立預測管道"""
    return PredictionPipeline(device=device, models_dir=models_dir)