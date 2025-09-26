"""
先進面部檢測器

使用多種演算法提高面部檢測準確率：
1. OpenCV Haar Cascade (傳統方法)
2. OpenCV DNN face detector (深度學習)
3. 多尺度檢測
4. 檢測結果融合
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class AdvancedFaceDetector:
    """先進面部檢測器"""

    def __init__(self):
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4

        # 初始化檢測器
        self.haar_detector = self._load_haar_detector()
        self.dnn_detector = self._load_dnn_detector()

        self.detection_methods = []
        if self.haar_detector:
            self.detection_methods.append('haar')
        if self.dnn_detector:
            self.detection_methods.append('dnn')

        logger.info(f"已初始化面部檢測器，可用方法: {self.detection_methods}")

    def _load_haar_detector(self) -> Optional[cv2.CascadeClassifier]:
        """載入Haar Cascade檢測器"""
        try:
            haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            detector = cv2.CascadeClassifier(haar_path)

            if detector.empty():
                logger.warning("Haar Cascade檢測器載入失敗")
                return None

            logger.info("Haar Cascade檢測器載入成功")
            return detector

        except Exception as e:
            logger.error(f"載入Haar檢測器失敗: {e}")
            return None

    def _load_dnn_detector(self) -> Optional[Tuple[cv2.dnn.Net, Tuple[int, int]]]:
        """載入DNN面部檢測器"""
        try:
            # 嘗試載入OpenCV預訓練的DNN模型
            # 這裡我們創建一個簡化版本，實際應用中可以下載預訓練模型

            # 檢查是否有可用的DNN模型檔案
            model_files = [
                "opencv_face_detector_uint8.pb",  # 模型檔案
                "opencv_face_detector.pbtxt"      # 配置檔案
            ]

            # 實際部署時，這些檔案應該從官方下載
            # 這裡我們只是示範架構，不載入實際模型
            logger.info("DNN面部檢測器架構已準備（需要模型檔案）")
            return None  # 暫時返回None，實際部署時載入模型

        except Exception as e:
            logger.error(f"載入DNN檢測器失敗: {e}")
            return None

    def detect_faces_haar(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """使用Haar Cascade檢測面部"""
        if not self.haar_detector:
            return []

        try:
            # 轉換為灰階
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # 多尺度檢測
            detections = []

            # 不同的檢測參數組合
            detection_configs = [
                # (scale_factor, min_neighbors, min_size)
                (1.1, 5, (50, 50)),
                (1.2, 4, (40, 40)),
                (1.3, 3, (30, 30)),
                (1.1, 3, (60, 60)),
            ]

            for scale_factor, min_neighbors, min_size in detection_configs:
                faces = self.haar_detector.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=min_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                for (x, y, w, h) in faces:
                    detections.append({
                        'bbox': (x, y, w, h),
                        'confidence': 0.8,  # Haar沒有置信度，使用固定值
                        'method': 'haar',
                        'area': w * h
                    })

            # 去重和排序
            detections = self._remove_duplicate_detections(detections)
            detections.sort(key=lambda x: x['area'], reverse=True)

            return detections

        except Exception as e:
            logger.error(f"Haar面部檢測失敗: {e}")
            return []

    def detect_faces_dnn(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """使用DNN檢測面部"""
        if not self.dnn_detector:
            return []

        try:
            # 這裡是DNN檢測的實現框架
            # 實際部署時需要載入預訓練模型

            logger.debug("DNN面部檢測（需要模型檔案）")
            return []

        except Exception as e:
            logger.error(f"DNN面部檢測失敗: {e}")
            return []

    def _remove_duplicate_detections(
        self,
        detections: List[Dict[str, Any]],
        iou_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """移除重複的檢測結果"""
        if not detections:
            return []

        # 按置信度排序
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        keep = []
        for detection in detections:
            bbox1 = detection['bbox']
            should_keep = True

            for kept_detection in keep:
                bbox2 = kept_detection['bbox']
                iou = self._calculate_iou(bbox1, bbox2)

                if iou > iou_threshold:
                    should_keep = False
                    break

            if should_keep:
                keep.append(detection)

        return keep

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """計算兩個邊界框的IoU"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # 計算交集
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def detect_faces_enhanced(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """增強版面部檢測 - 融合多種方法"""
        try:
            all_detections = []

            # 使用Haar檢測
            if 'haar' in self.detection_methods:
                haar_detections = self.detect_faces_haar(image)
                all_detections.extend(haar_detections)

            # 使用DNN檢測（如果可用）
            if 'dnn' in self.detection_methods:
                dnn_detections = self.detect_faces_dnn(image)
                all_detections.extend(dnn_detections)

            # 融合檢測結果
            final_detections = self._merge_detections(all_detections)

            # 品質過濾
            final_detections = self._filter_by_quality(final_detections, image)

            logger.debug(f"檢測到 {len(final_detections)} 個高品質面部")
            return final_detections

        except Exception as e:
            logger.error(f"增強面部檢測失敗: {e}")
            return []

    def _merge_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """融合不同方法的檢測結果"""
        if not detections:
            return []

        # 去重
        merged = self._remove_duplicate_detections(detections, iou_threshold=0.4)

        # 重新計算置信度（如果有多個方法檢測到同一區域）
        for detection in merged:
            bbox = detection['bbox']
            overlapping_detections = [
                d for d in detections
                if self._calculate_iou(bbox, d['bbox']) > 0.2
            ]

            if len(overlapping_detections) > 1:
                # 多個方法檢測到，提高置信度
                detection['confidence'] = min(0.95, detection['confidence'] + 0.2)
                detection['method'] = 'merged'

        return merged

    def _filter_by_quality(
        self,
        detections: List[Dict[str, Any]],
        image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """基於品質過濾檢測結果"""
        filtered = []

        for detection in detections:
            x, y, w, h = detection['bbox']

            # 檢查邊界
            if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
                continue

            # 檢查尺寸
            if w < 30 or h < 30:  # 太小
                continue

            if w > image.shape[1] * 0.8 or h > image.shape[0] * 0.8:  # 太大
                continue

            # 檢查比例
            ratio = w / h
            if ratio < 0.5 or ratio > 2.0:  # 比例異常
                continue

            # 檢查圖片品質
            face_region = image[y:y+h, x:x+w]
            if self._check_face_region_quality(face_region):
                filtered.append(detection)

        return filtered

    def _check_face_region_quality(self, face_region: np.ndarray) -> bool:
        """檢查面部區域品質"""
        try:
            if face_region.size == 0:
                return False

            # 轉換為灰階
            if len(face_region.shape) == 3:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
            else:
                gray_face = face_region

            # 檢查對比度
            std_dev = np.std(gray_face)
            if std_dev < 15:  # 對比度太低
                return False

            # 檢查亮度
            mean_brightness = np.mean(gray_face)
            if mean_brightness < 30 or mean_brightness > 220:  # 太暗或太亮
                return False

            # 檢查清晰度（使用拉普拉斯算子）
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            if laplacian_var < 50:  # 太模糊
                return False

            return True

        except Exception as e:
            logger.debug(f"面部區域品質檢查失敗: {e}")
            return False

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """主要的面部檢測接口"""
        return self.detect_faces_enhanced(image)

    def get_detection_summary(self, image: np.ndarray) -> Dict[str, Any]:
        """獲取檢測總結"""
        try:
            detections = self.detect_faces(image)

            return {
                'faces_detected': len(detections),
                'detectable': len(detections) > 0,
                'face_boxes': [d['bbox'] for d in detections],
                'confidence_scores': [d['confidence'] for d in detections],
                'methods_used': list(set(d['method'] for d in detections)),
                'largest_face_area': max([d['area'] for d in detections]) if detections else 0,
                'average_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0.0
            }

        except Exception as e:
            logger.error(f"檢測總結生成失敗: {e}")
            return {
                'faces_detected': 0,
                'detectable': False,
                'face_boxes': [],
                'confidence_scores': [],
                'methods_used': [],
                'largest_face_area': 0,
                'average_confidence': 0.0,
                'error': str(e)
            }


# 創建全局實例
advanced_face_detector = AdvancedFaceDetector()