"""
圖像服務層

處理圖像上傳、存儲、驗證和預處理業務邏輯
"""

import os
import uuid
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import asyncio
import json

from PIL import Image, ImageEnhance
import numpy as np
import cv2

from ..core.config import get_settings
from ..ml_pipeline.data_processing.image_processor import ImageProcessor
from ..ml_pipeline.models.face_detection.detector import FaceDetector

logger = logging.getLogger(__name__)
settings = get_settings()


class ImageService:
    """圖像服務類"""

    def __init__(self):
        self.settings = settings
        self.upload_dir = Path(settings.UPLOAD_TEMP_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        # 初始化圖像處理器
        self.image_processor = ImageProcessor()
        self.face_detector = FaceDetector(
            min_face_size=60,
            device="cpu"  # 默認使用CPU
        )

        # 內存中的圖像緩存（簡化版本）
        self._image_cache = {}
        self._user_images = {}  # user_id -> [image_info]

    async def process_uploaded_image(
        self,
        image: Image.Image,
        user_id: str,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """處理上傳的圖像"""
        try:
            start_time = datetime.utcnow()

            # 生成唯一圖像ID
            image_id = str(uuid.uuid4())

            # 獲取圖像基本資訊
            width, height = image.size
            image_info = {
                'width': width,
                'height': height,
                'format': image.format or 'JPEG',
                'mode': image.mode,
                'has_transparency': image.mode in ['RGBA', 'LA'] or 'transparency' in image.info
            }

            # 轉換為numpy陣列進行處理
            image_array = np.array(image)

            # 圖像質量檢查
            quality_check = await self._check_image_quality(image_array)
            if not quality_check['acceptable']:
                return {
                    'success': False,
                    'message': f"圖像質量不符合要求: {quality_check['reason']}",
                    'quality_score': quality_check['score']
                }

            # 面部檢測
            face_detection_result = await self._detect_faces(image_array)

            # 圖像增強處理
            enhanced_image = self.image_processor.enhance_image(image_array)

            # 計算圖像哈希值（用於去重檢查）
            image_hash = self._calculate_image_hash(image_array)

            # 檢查是否重複上傳
            if await self._is_duplicate_image(user_id, image_hash):
                return {
                    'success': False,
                    'message': '檢測到重複的圖像，請上傳不同的照片'
                }

            # 保存處理後的圖像（可選）
            saved_path = None
            if self.settings.ENVIRONMENT == "development":
                saved_path = await self._save_image_to_disk(
                    enhanced_image, image_id, image.format or 'JPEG'
                )

            # 創建圖像記錄
            image_record = {
                'image_id': image_id,
                'user_id': user_id,
                'filename': f"{image_id}.{image.format.lower() if image.format else 'jpg'}",
                'upload_time': start_time,
                'file_size': len(np.array(image).tobytes()),
                'width': width,
                'height': height,
                'format': image.format or 'JPEG',
                'description': description,
                'image_hash': image_hash,
                'face_detected': face_detection_result['faces_detected'] > 0,
                'face_count': face_detection_result['faces_detected'],
                'face_quality': face_detection_result.get('primary_face_quality', 0),
                'processing_metadata': {
                    'quality_score': quality_check['score'],
                    'enhancement_applied': True,
                    'face_detection_confidence': face_detection_result.get('primary_face_confidence', 0),
                    'saved_path': str(saved_path) if saved_path else None
                },
                'status': 'processed'
            }

            # 存儲到緩存
            self._image_cache[image_id] = {
                'record': image_record,
                'processed_image': enhanced_image,
                'face_data': face_detection_result
            }

            # 更新用戶圖像列表
            if user_id not in self._user_images:
                self._user_images[user_id] = []
            self._user_images[user_id].append(image_record)

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            logger.info(f"圖像處理完成: {image_id}, 用戶: {user_id}, 處理時間: {processing_time:.2f}s")

            return {
                'success': True,
                'message': '圖像處理成功',
                'image_id': image_id,
                'image_info': image_info,
                'face_detected': face_detection_result['faces_detected'] > 0,
                'face_count': face_detection_result['faces_detected'],
                'quality_score': quality_check['score'],
                'processing_time': processing_time
            }

        except Exception as e:
            logger.error(f"圖像處理失敗: {e}")
            return {
                'success': False,
                'message': f'圖像處理失敗: {str(e)}'
            }

    async def _check_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """檢查圖像質量"""
        try:
            # 基本尺寸檢查
            height, width = image.shape[:2]
            if height < 200 or width < 200:
                return {
                    'acceptable': False,
                    'reason': '圖像尺寸過小，建議至少200x200像素',
                    'score': 0.2
                }

            # 轉換為灰階進行分析
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

            # 模糊度檢測（拉普拉斯變異數）
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 500.0, 1.0)  # 正規化到0-1

            # 亮度檢測
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128.0  # 理想亮度128

            # 對比度檢測
            contrast = np.std(gray)
            contrast_score = min(contrast / 64.0, 1.0)  # 正規化到0-1

            # 噪點檢測（簡化版）
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise_level = np.std(gray - blurred)
            noise_score = max(1.0 - noise_level / 20.0, 0.0)

            # 綜合評分
            overall_score = (sharpness_score * 0.4 + brightness_score * 0.2 +
                           contrast_score * 0.2 + noise_score * 0.2)

            # 判斷是否可接受
            acceptable = overall_score >= 0.5
            reason = []

            if sharpness_score < 0.3:
                reason.append("圖像模糊")
            if brightness_score < 0.5:
                reason.append("亮度不適當")
            if contrast_score < 0.3:
                reason.append("對比度過低")
            if noise_score < 0.3:
                reason.append("噪點過多")

            return {
                'acceptable': acceptable,
                'score': round(overall_score, 3),
                'reason': ', '.join(reason) if reason else '圖像質量良好',
                'metrics': {
                    'sharpness': round(sharpness_score, 3),
                    'brightness': round(brightness_score, 3),
                    'contrast': round(contrast_score, 3),
                    'noise': round(noise_score, 3)
                }
            }

        except Exception as e:
            logger.error(f"圖像質量檢查失敗: {e}")
            return {
                'acceptable': False,
                'score': 0.0,
                'reason': '圖像質量檢查失敗'
            }

    async def _detect_faces(self, image: np.ndarray) -> Dict[str, Any]:
        """檢測圖像中的面部"""
        try:
            # 使用面部檢測器
            detection_result = self.face_detector.predict(image)

            if detection_result['success']:
                return {
                    'faces_detected': len(detection_result['faces']),
                    'primary_face_confidence': detection_result.get('primary_face', {}).get('confidence', 0),
                    'primary_face_quality': detection_result.get('detection_confidence', 0),
                    'face_regions': detection_result.get('face_regions', []),
                    'detection_metadata': {
                        'detector_version': self.face_detector.version,
                        'processing_time': 0  # TODO: 實際計算處理時間
                    }
                }
            else:
                return {
                    'faces_detected': 0,
                    'primary_face_confidence': 0,
                    'primary_face_quality': 0,
                    'face_regions': [],
                    'error': detection_result.get('message', '面部檢測失敗')
                }

        except Exception as e:
            logger.error(f"面部檢測失敗: {e}")
            return {
                'faces_detected': 0,
                'primary_face_confidence': 0,
                'primary_face_quality': 0,
                'face_regions': [],
                'error': f'檢測過程出錯: {str(e)}'
            }

    def _calculate_image_hash(self, image: np.ndarray) -> str:
        """計算圖像哈希值"""
        try:
            # 縮放到固定尺寸以提高哈希一致性
            resized = cv2.resize(image, (64, 64))

            # 轉換為灰階
            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            else:
                gray = resized

            # 計算感知哈希
            # 這裡使用簡化版本，實際應用可以使用更強的算法
            image_bytes = gray.tobytes()
            return hashlib.md5(image_bytes).hexdigest()

        except Exception as e:
            logger.error(f"計算圖像哈希失敗: {e}")
            return str(uuid.uuid4())  # 回退到隨機ID

    async def _is_duplicate_image(self, user_id: str, image_hash: str) -> bool:
        """檢查是否為重複圖像"""
        try:
            if user_id not in self._user_images:
                return False

            # 檢查用戶最近上傳的圖像
            recent_images = [
                img for img in self._user_images[user_id]
                if (datetime.utcnow() - img['upload_time']).days < 7  # 檢查最近7天
            ]

            for img in recent_images:
                if img.get('image_hash') == image_hash:
                    return True

            return False

        except Exception as e:
            logger.error(f"重複檢查失敗: {e}")
            return False

    async def _save_image_to_disk(self, image: np.ndarray, image_id: str, format: str) -> Optional[Path]:
        """保存圖像到磁碟（開發環境）"""
        try:
            if self.settings.ENVIRONMENT != "development":
                return None

            # 創建保存路徑
            filename = f"{image_id}.{format.lower()}"
            save_path = self.upload_dir / filename

            # 轉換為PIL圖像並保存
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image, 'RGB')
            else:
                pil_image = Image.fromarray(image, 'L')

            pil_image.save(save_path, format=format, quality=90, optimize=True)

            logger.debug(f"圖像已保存: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"保存圖像失敗: {e}")
            return None

    async def get_image_data(self, image_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """獲取圖像數據"""
        try:
            if image_id not in self._image_cache:
                return None

            image_data = self._image_cache[image_id]
            record = image_data['record']

            # 驗證權限
            if record['user_id'] != user_id:
                logger.warning(f"用戶 {user_id} 嘗試存取不屬於自己的圖像 {image_id}")
                return None

            return image_data

        except Exception as e:
            logger.error(f"獲取圖像數據失敗: {e}")
            return None

    async def get_user_history(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """獲取用戶的圖像上傳歷史"""
        try:
            if user_id not in self._user_images:
                return []

            # 按上傳時間排序
            user_images = sorted(
                self._user_images[user_id],
                key=lambda x: x['upload_time'],
                reverse=True
            )

            # 分頁
            start = offset
            end = offset + limit
            page_images = user_images[start:end]

            # 格式化返回數據
            history = []
            for img in page_images:
                history.append({
                    'image_id': img['image_id'],
                    'filename': img['filename'],
                    'upload_time': img['upload_time'],
                    'file_size': img['file_size'],
                    'width': img['width'],
                    'height': img['height'],
                    'format': img['format'],
                    'face_detected': img['face_detected'],
                    'has_prediction': False,  # TODO: 檢查是否有預測記錄
                    'description': img.get('description'),
                    'quality_score': img.get('processing_metadata', {}).get('quality_score', 0)
                })

            return history

        except Exception as e:
            logger.error(f"獲取用戶歷史失敗: {e}")
            return []

    async def delete_user_image(self, image_id: str, user_id: str) -> Dict[str, Any]:
        """刪除用戶圖像"""
        try:
            if image_id not in self._image_cache:
                return {
                    'success': False,
                    'message': '圖像不存在'
                }

            image_data = self._image_cache[image_id]
            record = image_data['record']

            # 驗證權限
            if record['user_id'] != user_id:
                return {
                    'success': False,
                    'message': '無權限刪除此圖像'
                }

            # 從緩存中刪除
            del self._image_cache[image_id]

            # 從用戶列表中刪除
            if user_id in self._user_images:
                self._user_images[user_id] = [
                    img for img in self._user_images[user_id]
                    if img['image_id'] != image_id
                ]

            # 刪除磁碟文件（如果存在）
            saved_path = record.get('processing_metadata', {}).get('saved_path')
            if saved_path and Path(saved_path).exists():
                try:
                    Path(saved_path).unlink()
                    logger.debug(f"已刪除磁碟文件: {saved_path}")
                except Exception as e:
                    logger.warning(f"刪除磁碟文件失敗: {e}")

            logger.info(f"圖像已刪除: {image_id}, 用戶: {user_id}")

            return {
                'success': True,
                'message': '圖像已刪除'
            }

        except Exception as e:
            logger.error(f"刪除圖像失敗: {e}")
            return {
                'success': False,
                'message': f'刪除失敗: {str(e)}'
            }

    def get_cache_stats(self) -> Dict[str, Any]:
        """獲取緩存統計資訊"""
        return {
            'total_cached_images': len(self._image_cache),
            'total_users': len(self._user_images),
            'cache_size_mb': len(str(self._image_cache)) / 1024 / 1024  # 簡化估算
        }