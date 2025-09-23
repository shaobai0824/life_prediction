"""
面部檢測模型

使用 MTCNN 進行面部檢測和對齊
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class FaceDetector(BaseModel):
    """面部檢測器"""

    def __init__(self, min_face_size: int = 40, device: str = None):
        super().__init__("FaceDetector", "1.0.0")

        self.min_face_size = min_face_size
        self.confidence_threshold = 0.7
        self.nms_threshold = 0.4

        if device:
            self.device = torch.device(device)

        # 初始化檢測器（這裡使用簡化版本）
        self._init_detector()

    def _init_detector(self):
        """初始化檢測器"""
        try:
            # 這裡應該載入真實的 MTCNN 模型
            # 現在使用 OpenCV 的 Haar Cascade 作為簡化版本
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            if self.face_cascade.empty():
                logger.error("無法載入面部檢測器")
                raise Exception("Face detector initialization failed")

            self.is_trained = True
            logger.info("面部檢測器初始化成功")

        except Exception as e:
            logger.error(f"檢測器初始化失敗: {e}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播（佔位符）"""
        # 實際的 MTCNN 前向傳播邏輯
        return x

    def preprocess(self, image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """預處理輸入圖像"""
        try:
            # 如果是檔案路徑，讀取圖像
            if isinstance(image, (str, Path)):
                image = cv2.imread(str(image))
                if image is None:
                    raise ValueError(f"無法讀取圖像: {image}")

            # 轉換為 RGB（OpenCV 預設是 BGR）
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 轉換為灰階用於檢測
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

            return gray

        except Exception as e:
            logger.error(f"圖像預處理失敗: {e}")
            raise

    def detect_faces(self, image: Union[np.ndarray, str, Path]) -> List[Dict[str, Any]]:
        """檢測面部"""
        try:
            # 預處理圖像
            gray = self.preprocess(image)

            # 檢測面部
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # 轉換結果格式
            detected_faces = []
            for i, (x, y, w, h) in enumerate(faces):
                face_info = {
                    'id': i,
                    'bbox': [int(x), int(y), int(w), int(h)],  # [x, y, width, height]
                    'confidence': 0.9,  # 簡化版本，使用固定信心度
                    'landmarks': self._detect_landmarks(gray, x, y, w, h),
                    'area': w * h,
                }
                detected_faces.append(face_info)

            # 按面積排序（最大的面部優先）
            detected_faces.sort(key=lambda x: x['area'], reverse=True)

            logger.info(f"檢測到 {len(detected_faces)} 個面部")
            return detected_faces

        except Exception as e:
            logger.error(f"面部檢測失敗: {e}")
            return []

    def _detect_landmarks(self, gray: np.ndarray, x: int, y: int, w: int, h: int) -> List[Tuple[int, int]]:
        """檢測面部關鍵點（簡化版本）"""
        # 這裡應該使用真實的關鍵點檢測模型
        # 現在返回一些基本的估計點
        landmarks = [
            (x + w // 4, y + h // 3),      # 左眼
            (x + 3 * w // 4, y + h // 3),  # 右眼
            (x + w // 2, y + 2 * h // 3),  # 鼻子
            (x + w // 3, y + 4 * h // 5),  # 左嘴角
            (x + 2 * w // 3, y + 4 * h // 5),  # 右嘴角
        ]
        return landmarks

    def extract_face_region(
        self,
        image: np.ndarray,
        face_info: Dict[str, Any],
        padding: float = 0.2,
        target_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """提取面部區域"""
        try:
            x, y, w, h = face_info['bbox']

            # 添加 padding
            pad_w = int(w * padding)
            pad_h = int(h * padding)

            # 計算擴展後的邊界框
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(image.shape[1], x + w + pad_w)
            y2 = min(image.shape[0], y + h + pad_h)

            # 提取面部區域
            face_region = image[y1:y2, x1:x2]

            # 調整大小
            if target_size:
                face_region = cv2.resize(face_region, target_size)

            return face_region

        except Exception as e:
            logger.error(f"提取面部區域失敗: {e}")
            return np.array([])

    def predict(self, input_data: Union[np.ndarray, str, Path]) -> Dict[str, Any]:
        """進行面部檢測預測"""
        try:
            # 檢測面部
            faces = self.detect_faces(input_data)

            if not faces:
                return {
                    'success': False,
                    'message': '未檢測到面部',
                    'faces': [],
                    'primary_face': None
                }

            # 獲取主要面部（面積最大的）
            primary_face = faces[0] if faces else None

            # 如果有原始圖像，提取面部區域
            face_regions = []
            if isinstance(input_data, np.ndarray) or isinstance(input_data, (str, Path)):
                if isinstance(input_data, (str, Path)):
                    image = cv2.imread(str(input_data))
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = input_data

                if image is not None:
                    for face in faces[:3]:  # 最多處理前3個面部
                        face_region = self.extract_face_region(image, face)
                        if face_region.size > 0:
                            face_regions.append(face_region)

            return {
                'success': True,
                'message': f'檢測到 {len(faces)} 個面部',
                'faces': faces,
                'primary_face': primary_face,
                'face_regions': face_regions,
                'detection_confidence': primary_face['confidence'] if primary_face else 0.0
            }

        except Exception as e:
            logger.error(f"面部檢測預測失敗: {e}")
            return {
                'success': False,
                'message': f'檢測失敗: {str(e)}',
                'faces': [],
                'primary_face': None
            }

    def postprocess(self, model_output: torch.Tensor) -> Dict[str, Any]:
        """後處理模型輸出"""
        # 實際的 MTCNN 後處理邏輯
        return {'processed': True}

    def validate_image(self, image: Union[np.ndarray, str, Path]) -> bool:
        """驗證圖像是否適合面部檢測"""
        try:
            if isinstance(image, (str, Path)):
                img_path = Path(image)
                if not img_path.exists():
                    logger.error(f"圖像檔案不存在: {img_path}")
                    return False

                # 檢查檔案大小（最大 10MB）
                if img_path.stat().st_size > 10 * 1024 * 1024:
                    logger.error("圖像檔案過大")
                    return False

                # 讀取圖像檢查
                test_img = cv2.imread(str(img_path))
                if test_img is None:
                    logger.error("無法讀取圖像檔案")
                    return False

                image = test_img

            elif isinstance(image, np.ndarray):
                # 檢查圖像尺寸
                if len(image.shape) < 2 or len(image.shape) > 3:
                    logger.error("不支援的圖像格式")
                    return False

                # 檢查最小尺寸
                if min(image.shape[:2]) < self.min_face_size:
                    logger.error("圖像尺寸太小")
                    return False

            else:
                logger.error("不支援的圖像類型")
                return False

            return True

        except Exception as e:
            logger.error(f"圖像驗證失敗: {e}")
            return False


def create_face_detector(min_face_size: int = 40, device: str = None) -> FaceDetector:
    """建立面部檢測器"""
    return FaceDetector(min_face_size=min_face_size, device=device)