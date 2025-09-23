"""
面部特徵提取模型

使用深度學習模型提取面部特徵用於壽命預測
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Union
from pathlib import Path
import logging

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class FaceFeatureExtractor(BaseModel):
    """面部特徵提取器"""

    def __init__(self, feature_dim: int = 512, device: str = None):
        super().__init__("FaceFeatureExtractor", "1.0.0")

        self.feature_dim = feature_dim
        if device:
            self.device = torch.device(device)

        # 建立特徵提取網路
        self._build_network()
        self.to(self.device)

    def _build_network(self):
        """建立特徵提取網路（基於 ResNet 的簡化版本）"""
        try:
            # 卷積層
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # 殘差塊
            self.layer1 = self._make_layer(64, 64, 2)
            self.layer2 = self._make_layer(64, 128, 2, stride=2)
            self.layer3 = self._make_layer(128, 256, 2, stride=2)
            self.layer4 = self._make_layer(256, 512, 2, stride=2)

            # 全域平均池化
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            # 特徵映射層
            self.feature_fc = nn.Sequential(
                nn.Linear(512, self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(self.feature_dim, self.feature_dim)
            )

            # 特徵分析層（用於解釋性）
            self.analysis_layers = nn.ModuleDict({
                'age_features': nn.Linear(self.feature_dim, 64),
                'health_features': nn.Linear(self.feature_dim, 64),
                'facial_structure': nn.Linear(self.feature_dim, 64),
                'skin_quality': nn.Linear(self.feature_dim, 64),
                'expression': nn.Linear(self.feature_dim, 32)
            })

            # 初始化權重
            self._initialize_weights()

            logger.info("特徵提取網路建立成功")

        except Exception as e:
            logger.error(f"網路建立失敗: {e}")
            raise

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """建立殘差層"""
        layers = []

        # 第一個塊可能需要下採樣
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        # 其他塊
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """初始化權重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播"""
        # 基礎卷積層
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 殘差層
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全域平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # 特徵映射
        features = self.feature_fc(x)

        return features

    def preprocess(self, face_image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """預處理面部圖像"""
        try:
            if isinstance(face_image, np.ndarray):
                # 確保是 RGB 格式
                if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                    # 正規化到 [0, 1]
                    image = face_image.astype(np.float32) / 255.0

                    # 調整大小到 224x224
                    if image.shape[:2] != (224, 224):
                        image = cv2.resize(image, (224, 224))

                    # 標準化 (ImageNet 統計)
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    image = (image - mean) / std

                    # 轉換為 tensor 並調整維度 (C, H, W)
                    tensor = torch.from_numpy(image).permute(2, 0, 1).float()

                    # 添加批次維度
                    if tensor.dim() == 3:
                        tensor = tensor.unsqueeze(0)

                else:
                    raise ValueError("圖像必須是 3 通道 RGB 格式")

            elif isinstance(face_image, torch.Tensor):
                tensor = face_image
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
            else:
                raise ValueError("不支援的輸入類型")

            return tensor.to(self.device)

        except Exception as e:
            logger.error(f"預處理失敗: {e}")
            raise

    def extract_features(self, face_image: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """提取面部特徵"""
        try:
            self.set_eval_mode()

            # 預處理
            input_tensor = self.preprocess(face_image)

            # 特徵提取
            with torch.no_grad():
                features = self.forward(input_tensor)

                # 分析不同類型的特徵
                analyzed_features = {}
                for feature_type, layer in self.analysis_layers.items():
                    analyzed_features[feature_type] = layer(features).cpu().numpy()

            # 主要特徵向量
            main_features = features.cpu().numpy()

            return {
                'main_features': main_features,
                'analyzed_features': analyzed_features,
                'feature_dim': self.feature_dim,
                'confidence': 0.95  # 簡化版本
            }

        except Exception as e:
            logger.error(f"特徵提取失敗: {e}")
            return {}

    def analyze_facial_characteristics(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """分析面部特徵（用於解釋性）"""
        try:
            analyzed = features.get('analyzed_features', {})

            # 年齡相關特徵分析
            age_score = np.mean(analyzed.get('age_features', [0])) if 'age_features' in analyzed else 0

            # 健康相關特徵分析
            health_score = np.mean(analyzed.get('health_features', [0])) if 'health_features' in analyzed else 0

            # 面部結構分析
            structure_score = np.mean(analyzed.get('facial_structure', [0])) if 'facial_structure' in analyzed else 0

            # 皮膚質量分析
            skin_score = np.mean(analyzed.get('skin_quality', [0])) if 'skin_quality' in analyzed else 0

            # 表情分析
            expression_score = np.mean(analyzed.get('expression', [0])) if 'expression' in analyzed else 0

            return {
                'age_indicator': float(age_score),
                'health_indicator': float(health_score),
                'facial_structure_score': float(structure_score),
                'skin_quality_score': float(skin_score),
                'expression_score': float(expression_score),
                'overall_assessment': {
                    'vitality': (health_score + skin_score) / 2,
                    'maturity': age_score,
                    'structural_balance': structure_score
                }
            }

        except Exception as e:
            logger.error(f"面部特徵分析失敗: {e}")
            return {}

    def predict(self, input_data: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """進行特徵提取預測"""
        try:
            # 提取特徵
            features = self.extract_features(input_data)

            if not features:
                return {
                    'success': False,
                    'message': '特徵提取失敗',
                    'features': None
                }

            # 分析面部特徵
            characteristics = self.analyze_facial_characteristics(features)

            return {
                'success': True,
                'message': '特徵提取成功',
                'features': features['main_features'],
                'characteristics': characteristics,
                'feature_dim': self.feature_dim,
                'confidence': features.get('confidence', 0.95)
            }

        except Exception as e:
            logger.error(f"特徵提取預測失敗: {e}")
            return {
                'success': False,
                'message': f'預測失敗: {str(e)}',
                'features': None
            }

    def postprocess(self, model_output: torch.Tensor) -> Dict[str, Any]:
        """後處理模型輸出"""
        try:
            # 正規化特徵向量
            normalized_features = F.normalize(model_output, p=2, dim=1)

            return {
                'features': normalized_features.cpu().numpy(),
                'feature_norm': torch.norm(model_output, p=2, dim=1).cpu().numpy()
            }

        except Exception as e:
            logger.error(f"後處理失敗: {e}")
            return {}


class ResidualBlock(nn.Module):
    """殘差塊"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def create_feature_extractor(feature_dim: int = 512, device: str = None) -> FaceFeatureExtractor:
    """建立面部特徵提取器"""
    return FaceFeatureExtractor(feature_dim=feature_dim, device=device)