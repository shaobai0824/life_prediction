"""
真實圖片特徵訓練器

使用真實的面部圖片特徵進行壽命預測模型訓練，
替換之前的模擬特徵，實現真正的圖片到壽命預測
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import json
import logging
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from PIL import Image
import cv2
import io

from .trainer import LifespanTrainer, LifePredictionModel
from ..models.face_detection.detector import FaceDetector
from ..models.feature_extraction.extractor import FaceFeatureExtractor
from ..data_processing.image_processor import ImageProcessor
from ..data_collection.image_downloader import RealImageDownloader

logger = logging.getLogger(__name__)


class RealImageDataset(Dataset):
    """真實圖片資料集"""

    def __init__(self, data_file: str, image_cache_dir: str = None):
        self.data_file = Path(data_file)
        self.image_cache_dir = Path(image_cache_dir) if image_cache_dir else Path("data/images/cache")
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化AI組件
        self.face_detector = FaceDetector()
        self.feature_extractor = FaceFeatureExtractor()
        self.image_processor = ImageProcessor()
        self.image_downloader = RealImageDownloader(str(self.image_cache_dir))

        # 載入資料
        self.data = []
        self.features = []
        self.labels = []

        self._load_data()
        asyncio.run(self._process_real_images())

    def _load_data(self):
        """載入訓練資料"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info(f"載入 {len(self.data)} 個訓練樣本")
        except Exception as e:
            logger.error(f"資料載入失敗: {e}")
            raise

    async def _process_real_images(self):
        """處理真實圖片並提取特徵"""
        logger.info("開始處理真實圖片並提取面部特徵...")

        valid_samples = 0

        # 準備下載請求
        download_requests = []
        for sample in self.data:
            download_requests.append({
                'url': sample['photo_url'],
                'person_name': sample['person_name'],
                'photo_date': sample.get('photo_date')
            })

        # 批次下載圖片
        async with aiohttp.ClientSession() as session:
            downloaded_images = await self.image_downloader.batch_download_images(session, download_requests)

        # 建立下載結果對應表
        url_to_download = {req['url']: img for req in download_requests for img in downloaded_images if img['original_url'] == req['url']}

        for i, sample in enumerate(self.data):
            try:
                photo_url = sample['photo_url']

                # 檢查是否成功下載
                if photo_url not in url_to_download:
                    logger.warning(f"跳過樣本 {i+1}: {sample['person_name']} - 圖片未下載或無面部")
                    continue

                download_result = url_to_download[photo_url]

                # 載入圖片
                image = self.image_downloader.load_cached_image_as_array(download_result['cache_path'])
                if image is None:
                    logger.warning(f"跳過樣本 {i+1}: 無法載入圖片")
                    continue

                # 檢測面部
                faces = self.face_detector.detect_faces(image)
                if not faces:
                    logger.warning(f"跳過樣本 {i+1}: {sample['person_name']} - 未檢測到面部")
                    continue

                # 使用第一個檢測到的面部
                face_info = faces[0]

                # 提取面部特徵
                face_features = self.feature_extractor.extract_features(
                    image, face_info['bbox']
                )

                if face_features is None:
                    logger.warning(f"跳過樣本 {i+1}: {sample['person_name']} - 特徵提取失敗")
                    continue

                # 保存特徵和標籤
                self.features.append(face_features)
                self.labels.append(sample['remaining_lifespan_years'])
                valid_samples += 1

                if valid_samples % 5 == 0:
                    logger.info(f"已處理 {valid_samples} 個有效樣本")

            except Exception as e:
                logger.error(f"處理樣本 {i+1} 時發生錯誤: {e}")
                continue

        # 轉換為numpy陣列
        if self.features:
            self.features = np.array(self.features)
            self.labels = np.array(self.labels)
            logger.info(f"成功提取 {valid_samples} 個樣本的真實面部特徵")
            logger.info(f"特徵維度: {self.features.shape}")
        else:
            logger.error("沒有成功提取任何面部特徵！")
            raise ValueError("特徵提取失敗，無有效訓練資料")

    def _get_image(self, sample: Dict[str, Any]) -> Optional[np.ndarray]:
        """獲取圖片（從快取或下載）"""
        try:
            photo_url = sample['photo_url']

            # 創建快取檔案名
            url_hash = hash(photo_url) % (2**32)
            cache_file = self.image_cache_dir / f"{sample['person_name'].replace(' ', '_')}_{sample['photo_date'][:4]}_{url_hash}.jpg"

            # 檢查快取
            if cache_file.exists():
                image = cv2.imread(str(cache_file))
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 嘗試下載圖片（但因為是模擬URL會失敗）
            # 在這裡我們使用佔位符圖片來示範
            return self._create_placeholder_face_image(sample)

        except Exception as e:
            logger.error(f"獲取圖片失敗: {e}")
            return None

    def _create_placeholder_face_image(self, sample: Dict[str, Any]) -> np.ndarray:
        """創建佔位符面部圖片用於測試"""
        # 創建一個更真實的模擬面部圖片，讓Haar Cascade能夠檢測到

        # 基於人名創建一致的模擬面部特徵
        np.random.seed(hash(sample['person_name']) % (2**32))

        # 創建基礎面部輪廓 - 使用膚色
        image = np.ones((400, 300, 3), dtype=np.uint8) * 220

        # 畫一個更像真實面部的橢圓形輪廓
        center = (150, 200)
        axes = (80, 100)
        cv2.ellipse(image, center, axes, 0, 0, 360, (200, 180, 160), -1)

        # 眼睛區域 - 更像真實的眼睛
        # 左眼
        cv2.ellipse(image, (120, 170), (15, 8), 0, 0, 360, (255, 255, 255), -1)  # 眼白
        cv2.circle(image, (120, 170), 6, (50, 50, 50), -1)  # 瞳孔
        cv2.circle(image, (120, 170), 2, (0, 0, 0), -1)  # 黑瞳

        # 右眼
        cv2.ellipse(image, (180, 170), (15, 8), 0, 0, 360, (255, 255, 255), -1)  # 眼白
        cv2.circle(image, (180, 170), 6, (50, 50, 50), -1)  # 瞳孔
        cv2.circle(image, (180, 170), 2, (0, 0, 0), -1)  # 黑瞳

        # 眉毛
        cv2.ellipse(image, (120, 155), (20, 5), 0, 0, 180, (80, 60, 40), 3)
        cv2.ellipse(image, (180, 155), (20, 5), 0, 0, 180, (80, 60, 40), 3)

        # 鼻子 - 更立體的鼻子
        pts = np.array([[150, 180], [140, 210], [150, 220], [160, 210]], np.int32)
        cv2.fillPoly(image, [pts], (190, 170, 150))

        # 鼻孔
        cv2.ellipse(image, (145, 215), (3, 2), 0, 0, 360, (120, 100, 80), -1)
        cv2.ellipse(image, (155, 215), (3, 2), 0, 0, 360, (120, 100, 80), -1)

        # 嘴巴 - 更像真實的嘴巴
        cv2.ellipse(image, (150, 250), (25, 8), 0, 0, 360, (160, 80, 80), -1)
        cv2.line(image, (125, 250), (175, 250), (120, 60, 60), 2)

        # 基於年齡調整特徵
        age = sample['age_at_photo']
        if age > 50:
            # 添加皺紋線條
            cv2.line(image, (100, 180), (110, 175), (160, 140, 120), 1)
            cv2.line(image, (190, 175), (200, 180), (160, 140, 120), 1)
            # 額頭皺紋
            cv2.line(image, (110, 130), (190, 130), (160, 140, 120), 1)
            cv2.line(image, (115, 120), (185, 120), (160, 140, 120), 1)

        # 基於職業調整特徵
        occupation = sample.get('occupation', '')
        if 'Actor' in occupation or 'Actress' in occupation:
            # 演員通常保養較好，皮膚更光滑
            image = cv2.GaussianBlur(image, (3, 3), 0)
            image = cv2.addWeighted(image, 0.9, np.ones_like(image) * 230, 0.1, 0)
        elif 'Politician' in occupation:
            # 政治家可能較為嚴肅，添加一些陰影
            cv2.ellipse(image, (150, 200), (70, 90), 0, 45, 135, (180, 160, 140), 2)

        # 添加一些噪聲使其更真實
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)

        return image

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.labels[idx]])

    def get_statistics(self) -> Dict[str, Any]:
        """獲取資料統計"""
        if len(self.features) == 0:
            return {'error': 'No valid features extracted'}

        return {
            'num_samples': len(self.features),
            'feature_dim': self.features.shape[1],
            'label_stats': {
                'mean': float(np.mean(self.labels)),
                'std': float(np.std(self.labels)),
                'min': float(np.min(self.labels)),
                'max': float(np.max(self.labels))
            }
        }


class RealImageLifespanTrainer(LifespanTrainer):
    """真實圖片壽命預測訓練器"""

    def __init__(self, model: LifePredictionModel, config: Dict[str, Any] = None):
        super().__init__(model, config)

    def train_with_real_images(self, dataset: RealImageDataset) -> Dict[str, Any]:
        """使用真實圖片資料進行訓練"""
        try:
            logger.info("開始使用真實面部特徵進行模型訓練...")

            # 檢查資料集有效性
            stats = dataset.get_statistics()
            if 'error' in stats:
                return {
                    'success': False,
                    'error': '資料集無效，無法進行訓練'
                }

            logger.info(f"真實圖片資料集統計: {stats}")

            # 準備資料載入器
            total_size = len(dataset)
            val_size = int(total_size * self.config['val_split'])
            train_size = total_size - val_size

            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False
            )

            # 訓練循環
            patience_counter = 0

            for epoch in range(self.config['num_epochs']):
                # 訓練階段
                train_loss, train_mae = self._train_epoch(train_loader)

                # 驗證階段
                val_loss, val_mae = self._validate_epoch(val_loader)

                # 記錄歷史
                self.train_history['train_loss'].append(train_loss)
                self.train_history['val_loss'].append(val_loss)
                self.train_history['train_mae'].append(train_mae)
                self.train_history['val_mae'].append(val_mae)

                # 學習率調度
                self.scheduler.step(val_loss)

                # 早停檢查
                if val_loss < self.best_val_loss - self.config['min_delta']:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                # 輸出訓練進度
                if epoch % 5 == 0 or epoch == self.config['num_epochs'] - 1:
                    logger.info(
                        f"Epoch [{epoch+1}/{self.config['num_epochs']}] "
                        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                        f"Train MAE: {train_mae:.2f}, Val MAE: {val_mae:.2f}"
                    )

                # 早停
                if patience_counter >= self.config['patience']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            # 載入最佳模型
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)

            # 評估最終模型
            final_metrics = self._evaluate_model(val_loader, dataset)

            logger.info("真實圖片模型訓練完成!")
            logger.info(f"最佳驗證損失: {self.best_val_loss:.4f}")

            return {
                'success': True,
                'best_val_loss': self.best_val_loss,
                'train_history': self.train_history,
                'final_metrics': final_metrics,
                'config': self.config,
                'data_stats': stats
            }

        except Exception as e:
            logger.error(f"真實圖片模型訓練失敗: {e}")
            return {
                'success': False,
                'error': str(e)
            }


def train_with_real_images(
    data_file: str,
    output_dir: str = "models/trained",
    image_cache_dir: str = "data/images/cache",
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """使用真實圖片訓練壽命預測模型的便捷函數"""
    try:
        logger.info("準備使用真實圖片進行壽命預測模型訓練...")

        # 準備真實圖片資料集
        dataset = RealImageDataset(data_file, image_cache_dir)

        # 檢查資料集
        stats = dataset.get_statistics()
        if 'error' in stats:
            return {
                'success': False,
                'error': f'資料集準備失敗: {stats["error"]}'
            }

        logger.info(f"真實圖片資料集統計: {stats}")

        # 建立模型（根據實際提取的特徵維度）
        feature_dim = stats['feature_dim']
        model = LifePredictionModel(input_dim=feature_dim)

        # 建立訓練器
        trainer = RealImageLifespanTrainer(model, config)

        # 執行訓練
        training_result = trainer.train_with_real_images(dataset)

        if training_result['success']:
            # 保存模型
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            model_path = output_path / f"real_image_lifespan_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            trainer.save_model(str(model_path))

            training_result['model_path'] = str(model_path)

        return training_result

    except Exception as e:
        logger.error(f"真實圖片模型訓練失敗: {e}")
        return {
            'success': False,
            'error': str(e)
        }