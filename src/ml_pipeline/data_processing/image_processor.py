"""
圖像處理模組

提供圖像增強、正規化和預處理功能
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """圖像處理器"""

    def __init__(self):
        self.default_size = (224, 224)
        self.enhancement_params = {
            'brightness_factor': 1.1,
            'contrast_factor': 1.2,
            'saturation_factor': 1.1,
            'gaussian_blur_kernel': (3, 3),
            'sharpen_strength': 0.5
        }

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """增強圖像質量"""
        try:
            enhanced = image.copy()

            # 1. 降噪
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

            # 2. 對比度增強
            enhanced = self._adjust_contrast(enhanced, self.enhancement_params['contrast_factor'])

            # 3. 亮度調整
            enhanced = self._adjust_brightness(enhanced, self.enhancement_params['brightness_factor'])

            # 4. 銳化處理
            enhanced = self._sharpen_image(enhanced, self.enhancement_params['sharpen_strength'])

            # 5. 色彩飽和度調整
            enhanced = self._adjust_saturation(enhanced, self.enhancement_params['saturation_factor'])

            return enhanced

        except Exception as e:
            logger.warning(f"圖像增強失敗，返回原圖: {e}")
            return image

    def _adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """調整對比度"""
        try:
            # 轉換為 float32 進行計算
            img_float = image.astype(np.float32)

            # 計算平均亮度
            mean = np.mean(img_float)

            # 應用對比度調整
            enhanced = (img_float - mean) * factor + mean

            # 限制值範圍並轉回 uint8
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

            return enhanced

        except Exception as e:
            logger.warning(f"對比度調整失敗: {e}")
            return image

    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """調整亮度"""
        try:
            # 轉換為 HSV 顏色空間
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.float32)

            # 調整 V 通道（亮度）
            hsv[:, :, 2] = hsv[:, :, 2] * factor

            # 限制值範圍
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

            # 轉回 RGB
            hsv = hsv.astype(np.uint8)
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            return enhanced

        except Exception as e:
            logger.warning(f"亮度調整失敗: {e}")
            return image

    def _adjust_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """調整飽和度"""
        try:
            # 轉換為 HSV 顏色空間
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.float32)

            # 調整 S 通道（飽和度）
            hsv[:, :, 1] = hsv[:, :, 1] * factor

            # 限制值範圍
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

            # 轉回 RGB
            hsv = hsv.astype(np.uint8)
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            return enhanced

        except Exception as e:
            logger.warning(f"飽和度調整失敗: {e}")
            return image

    def _sharpen_image(self, image: np.ndarray, strength: float) -> np.ndarray:
        """銳化圖像"""
        try:
            # 銳化核
            kernel = np.array([
                [-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1]
            ]) * strength

            # 應用銳化
            sharpened = cv2.filter2D(image, -1, kernel)

            # 混合原圖和銳化圖
            enhanced = cv2.addWeighted(image, 1 - strength, sharpened, strength, 0)

            return enhanced

        except Exception as e:
            logger.warning(f"圖像銳化失敗: {e}")
            return image

    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int],
                    maintain_aspect: bool = True) -> np.ndarray:
        """調整圖像大小"""
        try:
            if maintain_aspect:
                # 保持寬高比
                h, w = image.shape[:2]
                target_w, target_h = target_size

                # 計算縮放比例
                scale = min(target_w / w, target_h / h)

                # 計算新尺寸
                new_w = int(w * scale)
                new_h = int(h * scale)

                # 調整大小
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

                # 創建目標尺寸的畫布
                canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

                # 計算置中位置
                start_x = (target_w - new_w) // 2
                start_y = (target_h - new_h) // 2

                # 將調整後的圖像放置到畫布中央
                canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized

                return canvas

            else:
                # 直接調整到目標尺寸
                return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

        except Exception as e:
            logger.error(f"圖像尺寸調整失敗: {e}")
            return image

    def normalize_image(self, image: np.ndarray, method: str = 'imagenet') -> np.ndarray:
        """正規化圖像"""
        try:
            # 轉換為 float32
            img_float = image.astype(np.float32) / 255.0

            if method == 'imagenet':
                # ImageNet 統計數據
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                normalized = (img_float - mean) / std

            elif method == 'zero_one':
                # 0-1 正規化
                normalized = img_float

            elif method == 'minus_one_one':
                # -1 到 1 正規化
                normalized = (img_float - 0.5) * 2

            elif method == 'standardize':
                # 標準化
                mean = np.mean(img_float, axis=(0, 1))
                std = np.std(img_float, axis=(0, 1))
                normalized = (img_float - mean) / (std + 1e-8)

            else:
                logger.warning(f"未知的正規化方法: {method}，使用 imagenet")
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                normalized = (img_float - mean) / std

            return normalized

        except Exception as e:
            logger.error(f"圖像正規化失敗: {e}")
            return image

    def detect_face_quality(self, face_image: np.ndarray) -> dict:
        """檢測面部圖像質量"""
        try:
            quality_metrics = {}

            # 1. 模糊度檢測
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_metrics['sharpness'] = float(laplacian_var)

            # 2. 亮度檢測
            brightness = np.mean(gray)
            quality_metrics['brightness'] = float(brightness)

            # 3. 對比度檢測
            contrast = np.std(gray)
            quality_metrics['contrast'] = float(contrast)

            # 4. 噪聲檢測（簡化版本）
            noise_level = np.std(cv2.medianBlur(gray, 5) - gray)
            quality_metrics['noise_level'] = float(noise_level)

            # 5. 整體質量評分
            quality_score = self._calculate_quality_score(quality_metrics)
            quality_metrics['overall_score'] = quality_score

            return quality_metrics

        except Exception as e:
            logger.error(f"面部質量檢測失敗: {e}")
            return {'overall_score': 0.5}  # 預設中等質量

    def _calculate_quality_score(self, metrics: dict) -> float:
        """計算整體質量分數"""
        try:
            # 設定理想值和權重
            ideal_values = {
                'sharpness': 100,
                'brightness': 128,
                'contrast': 50,
                'noise_level': 5
            }

            weights = {
                'sharpness': 0.4,
                'brightness': 0.3,
                'contrast': 0.2,
                'noise_level': 0.1
            }

            total_score = 0
            for metric, value in metrics.items():
                if metric in ideal_values:
                    ideal = ideal_values[metric]
                    weight = weights[metric]

                    # 計算偏差分數（越接近理想值分數越高）
                    if metric == 'noise_level':
                        # 噪聲越低越好
                        score = max(0, 1 - (value / (ideal * 2)))
                    else:
                        # 其他指標接近理想值最好
                        deviation = abs(value - ideal) / ideal
                        score = max(0, 1 - deviation)

                    total_score += score * weight

            return float(np.clip(total_score, 0, 1))

        except Exception as e:
            logger.error(f"質量分數計算失敗: {e}")
            return 0.5

    def preprocess_for_model(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """為模型預測預處理圖像"""
        try:
            # 使用預設大小如果沒有指定
            if target_size is None:
                target_size = self.default_size

            # 1. 增強圖像
            enhanced = self.enhance_image(image)

            # 2. 調整大小
            resized = self.resize_image(enhanced, target_size, maintain_aspect=True)

            # 3. 正規化
            normalized = self.normalize_image(resized, method='imagenet')

            return normalized

        except Exception as e:
            logger.error(f"模型預處理失敗: {e}")
            # 返回基本處理的圖像
            try:
                resized = cv2.resize(image, target_size or self.default_size)
                return resized.astype(np.float32) / 255.0
            except:
                return image

    def batch_process_images(self, images: list, target_size: Tuple[int, int] = None) -> list:
        """批次處理多張圖像"""
        processed_images = []

        for i, image in enumerate(images):
            try:
                processed = self.preprocess_for_model(image, target_size)
                processed_images.append(processed)
            except Exception as e:
                logger.error(f"批次處理第 {i} 張圖像失敗: {e}")
                # 添加空圖像或跳過
                processed_images.append(None)

        return processed_images