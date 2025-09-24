"""
模型訓練管理器

負責管理壽命預測模型的訓練流程，包括資料載入、預處理、訓練和評估
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
# import matplotlib.pyplot as plt  # 註解掉以避免編譯問題
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

from ..models.life_prediction.predictor import LifePredictionModel
from ..data_processing.image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class LifespanDataset:
    """壽命預測資料集"""

    def __init__(self, data_file: str, image_processor: ImageProcessor = None):
        self.data_file = Path(data_file)
        self.image_processor = image_processor or ImageProcessor()
        self.data = []
        self.features = None
        self.labels = None
        self.scaler = StandardScaler()

        self._load_data()
        self._extract_features()

    def _load_data(self):
        """載入訓練資料"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info(f"載入 {len(self.data)} 個訓練樣本")
        except Exception as e:
            logger.error(f"資料載入失敗: {e}")
            raise

    def _extract_features(self):
        """提取特徵和標籤"""
        try:
            features = []
            labels = []

            for sample in self.data:
                # 模擬特徵提取（實際應用中會從照片中提取面部特徵）
                feature = self._create_mock_features(sample)
                features.append(feature)

                # 標籤是剩餘壽命
                remaining_years = sample['remaining_lifespan_years']
                labels.append(remaining_years)

            # 轉換為numpy陣列
            self.features = np.array(features, dtype=np.float32)
            self.labels = np.array(labels, dtype=np.float32)

            # 正規化特徵
            self.features = self.scaler.fit_transform(self.features)

            logger.info(f"特徵維度: {self.features.shape}, 標籤維度: {self.labels.shape}")

        except Exception as e:
            logger.error(f"特徵提取失敗: {e}")
            raise

    def _create_mock_features(self, sample: Dict[str, Any]) -> np.ndarray:
        """創建模擬特徵（實際應用中會從照片中提取）"""
        # 基於可用資訊創建512維特徵向量
        feature_vector = np.random.randn(512).astype(np.float32)

        # 根據已知資訊調整部分特徵
        age = sample['age_at_photo']
        total_lifespan = sample['total_lifespan']
        nationality = sample['nationality']
        occupation = sample['occupation']

        # 年齡相關特徵 (前100維)
        age_normalized = age / 100.0
        feature_vector[:100] *= (1 + age_normalized * 0.5)

        # 壽命相關特徵 (101-200維)
        lifespan_normalized = total_lifespan / 100.0
        feature_vector[100:200] *= (1 + lifespan_normalized * 0.3)

        # 國籍編碼 (201-250維)
        nationality_hash = hash(nationality) % 50
        feature_vector[200 + nationality_hash] *= 2.0

        # 職業編碼 (251-300維)
        occupation_hash = hash(occupation) % 50
        feature_vector[250 + occupation_hash] *= 2.0

        # 添加一些隨機性但保持一致性
        np.random.seed(hash(sample['person_name']) % (2**32))
        noise = np.random.randn(512) * 0.1
        feature_vector += noise

        return feature_vector

    def get_torch_dataset(self) -> TensorDataset:
        """轉換為PyTorch資料集"""
        features_tensor = torch.FloatTensor(self.features)
        labels_tensor = torch.FloatTensor(self.labels)
        return TensorDataset(features_tensor, labels_tensor)

    def get_statistics(self) -> Dict[str, Any]:
        """獲取資料統計"""
        return {
            'num_samples': len(self.data),
            'feature_dim': self.features.shape[1] if self.features is not None else 0,
            'label_stats': {
                'mean': float(np.mean(self.labels)) if self.labels is not None else 0,
                'std': float(np.std(self.labels)) if self.labels is not None else 0,
                'min': float(np.min(self.labels)) if self.labels is not None else 0,
                'max': float(np.max(self.labels)) if self.labels is not None else 0
            }
        }


class LifespanTrainer:
    """壽命預測模型訓練器"""

    def __init__(self, model: LifePredictionModel, config: Dict[str, Any] = None):
        self.model = model
        self.config = config or self._get_default_config()
        self.device = self.model.device

        # 訓練歷史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }

        # 最佳模型
        self.best_val_loss = float('inf')
        self.best_model_state = None

        # 設定損失函數和優化器
        self._setup_training()

    def _get_default_config(self) -> Dict[str, Any]:
        """預設訓練配置"""
        return {
            'batch_size': 16,
            'num_epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'patience': 15,
            'min_delta': 0.001,
            'val_split': 0.2,
            'save_best_model': True,
            'plot_training': True
        }

    def _setup_training(self):
        """設定訓練元件"""
        # 複合損失函數
        self.criterion = self._create_loss_function()

        # 優化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        # 學習率調度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=10
        )

    def _create_loss_function(self):
        """創建複合損失函數"""
        def combined_loss(outputs, targets):
            # 主要預測損失
            main_pred = outputs['life_prediction'].squeeze()
            mse_loss = F.mse_loss(main_pred, targets)

            # 不確定性正規化
            uncertainty = outputs['uncertainty'].squeeze()
            uncertainty_reg = torch.mean(uncertainty) * 0.1

            # 範圍約束
            range_penalty = torch.mean(torch.clamp(main_pred - 120, min=0)) * 10
            range_penalty += torch.mean(torch.clamp(-main_pred, min=0)) * 10

            return mse_loss + uncertainty_reg + range_penalty

        return combined_loss

    def train(self, dataset: LifespanDataset) -> Dict[str, Any]:
        """執行模型訓練"""
        try:
            logger.info("開始模型訓練...")

            # 準備資料
            torch_dataset = dataset.get_torch_dataset()
            train_dataset, val_dataset = self._split_dataset(torch_dataset)

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
                if epoch % 10 == 0 or epoch == self.config['num_epochs'] - 1:
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

            logger.info("模型訓練完成!")
            logger.info(f"最佳驗證損失: {self.best_val_loss:.4f}")

            return {
                'success': True,
                'best_val_loss': self.best_val_loss,
                'train_history': self.train_history,
                'final_metrics': final_metrics,
                'config': self.config
            }

        except Exception as e:
            logger.error(f"模型訓練失敗: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _split_dataset(self, dataset: TensorDataset) -> Tuple[TensorDataset, TensorDataset]:
        """分割資料集"""
        total_size = len(dataset)
        val_size = int(total_size * self.config['val_split'])
        train_size = total_size - val_size

        return random_split(dataset, [train_size, val_size])

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """訓練一個epoch"""
        self.model.train()
        total_loss = 0
        total_mae = 0
        num_batches = 0

        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            # 前向傳播
            self.optimizer.zero_grad()
            outputs = self.model(features)

            # 計算損失
            loss = self.criterion(outputs, labels)

            # 反向傳播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 統計
            total_loss += loss.item()
            with torch.no_grad():
                pred = outputs['life_prediction'].squeeze()
                mae = torch.mean(torch.abs(pred - labels))
                total_mae += mae.item()

            num_batches += 1

        return total_loss / num_batches, total_mae / num_batches

    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """驗證一個epoch"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        num_batches = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                pred = outputs['life_prediction'].squeeze()
                mae = torch.mean(torch.abs(pred - labels))

                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1

        return total_loss / num_batches, total_mae / num_batches

    def _evaluate_model(self, val_loader: DataLoader, dataset: LifespanDataset) -> Dict[str, Any]:
        """評估模型性能"""
        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                pred = outputs['life_prediction'].squeeze()

                predictions.extend(pred.cpu().numpy())
                targets.extend(labels.cpu().numpy())

        predictions = np.array(predictions)
        targets = np.array(targets)

        # 計算評估指標
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)

        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'num_samples': len(predictions)
        }

    def save_model(self, save_path: str, include_history: bool = True):
        """保存模型"""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # 準備保存資料
            save_data = {
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'input_dim': self.model.input_dim,
                    'hidden_dims': self.model.hidden_dims,
                    'model_name': self.model.model_name,
                    'version': self.model.version
                },
                'training_config': self.config,
                'timestamp': datetime.now().isoformat()
            }

            if include_history:
                save_data['train_history'] = self.train_history

            torch.save(save_data, save_path)
            logger.info(f"模型已保存至: {save_path}")

        except Exception as e:
            logger.error(f"模型保存失敗: {e}")
            raise

    def plot_training_history(self, save_path: str = None):
        """繪製訓練歷史"""
        try:
            if not HAS_MATPLOTLIB:
                logger.warning("matplotlib 未安裝，無法繪製訓練歷史圖表")
                return

            fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 5))

            # 損失曲線
            epochs = range(1, len(self.train_history['train_loss']) + 1)
            ax1.plot(epochs, self.train_history['train_loss'], label='Train Loss')
            ax1.plot(epochs, self.train_history['val_loss'], label='Val Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)

            # MAE曲線
            ax2.plot(epochs, self.train_history['train_mae'], label='Train MAE')
            ax2.plot(epochs, self.train_history['val_mae'], label='Val MAE')
            ax2.set_title('Training and Validation MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE (years)')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                logger.info(f"訓練歷史圖表已保存至: {save_path}")
            else:
                plt.show()

        except Exception as e:
            logger.error(f"繪製訓練歷史失敗: {e}")


def load_model(model_path: str) -> Tuple[LifePredictionModel, Dict[str, Any]]:
    """載入已訓練的模型"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')

        # 重建模型
        model_config = checkpoint['model_config']
        model = LifePredictionModel(
            input_dim=model_config['input_dim'],
            hidden_dims=model_config['hidden_dims']
        )

        # 載入權重
        model.load_state_dict(checkpoint['model_state_dict'])

        # 載入配置
        training_config = checkpoint.get('training_config', {})

        logger.info(f"模型已從 {model_path} 載入")

        return model, training_config

    except Exception as e:
        logger.error(f"模型載入失敗: {e}")
        raise


# 方便函數
def train_lifespan_model(
    data_file: str,
    output_dir: str = "models/trained",
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """訓練壽命預測模型的便捷函數"""
    try:
        # 準備資料
        dataset = LifespanDataset(data_file)
        logger.info(f"資料集統計: {dataset.get_statistics()}")

        # 建立模型
        model = LifePredictionModel(input_dim=512)

        # 建立訓練器
        trainer = LifespanTrainer(model, config)

        # 執行訓練
        training_result = trainer.train(dataset)

        if training_result['success']:
            # 保存模型
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            model_path = output_path / f"lifespan_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            trainer.save_model(str(model_path))

            # 保存訓練圖表
            if config is None or config.get('plot_training', True):
                plot_path = output_path / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                trainer.plot_training_history(str(plot_path))

            training_result['model_path'] = str(model_path)

        return training_result

    except Exception as e:
        logger.error(f"模型訓練失敗: {e}")
        return {
            'success': False,
            'error': str(e)
        }