"""
基礎模型類別

提供所有 AI 模型的共用介面和功能
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
import json
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseModel(ABC, nn.Module):
    """所有 AI 模型的基礎類別"""

    def __init__(self, model_name: str, version: str = "1.0.0"):
        super().__init__()
        self.model_name = model_name
        self.version = version
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_trained = False
        self.training_history = []

        logger.info(f"初始化模型: {model_name} v{version}")
        logger.info(f"使用設備: {self.device}")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播"""
        pass

    @abstractmethod
    def predict(self, input_data: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """進行預測"""
        pass

    @abstractmethod
    def preprocess(self, raw_data: Any) -> torch.Tensor:
        """資料預處理"""
        pass

    @abstractmethod
    def postprocess(self, model_output: torch.Tensor) -> Dict[str, Any]:
        """後處理模型輸出"""
        pass

    def load_model(self, model_path: Union[str, Path]) -> bool:
        """載入模型權重"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"模型檔案不存在: {model_path}")
                return False

            # 載入模型權重
            checkpoint = torch.load(model_path, map_location=self.device)

            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
                self.version = checkpoint.get('version', self.version)
                self.is_trained = True

                logger.info(f"成功載入模型: {model_path}")
                logger.info(f"模型版本: {self.version}")

                return True
            else:
                logger.error("模型檔案格式不正確")
                return False

        except Exception as e:
            logger.error(f"載入模型失敗: {e}")
            return False

    def save_model(self, save_path: Union[str, Path], include_optimizer: bool = False) -> bool:
        """儲存模型"""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # 準備儲存的資料
            save_data = {
                'model_state_dict': self.state_dict(),
                'model_name': self.model_name,
                'version': self.version,
                'device': str(self.device),
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'save_time': datetime.now().isoformat(),
            }

            # 儲存模型
            torch.save(save_data, save_path)
            logger.info(f"模型已儲存: {save_path}")

            return True

        except Exception as e:
            logger.error(f"儲存模型失敗: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型資訊"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': self.model_name,
            'version': self.version,
            'device': str(self.device),
            'is_trained': self.is_trained,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2),  # 假設 float32
        }

    def validate_input(self, input_data: Any) -> bool:
        """驗證輸入資料"""
        try:
            if isinstance(input_data, np.ndarray):
                input_data = torch.from_numpy(input_data).float()

            if not isinstance(input_data, torch.Tensor):
                logger.error("輸入資料必須是 torch.Tensor 或 numpy.ndarray")
                return False

            if input_data.numel() == 0:
                logger.error("輸入資料不能為空")
                return False

            return True

        except Exception as e:
            logger.error(f"輸入驗證失敗: {e}")
            return False

    def to_device(self, data: torch.Tensor) -> torch.Tensor:
        """將資料移到指定設備"""
        return data.to(self.device)

    def set_eval_mode(self):
        """設定評估模式"""
        self.eval()
        torch.set_grad_enabled(False)

    def set_train_mode(self):
        """設定訓練模式"""
        self.train()
        torch.set_grad_enabled(True)

    def estimate_inference_time(self, input_shape: Tuple[int, ...], num_runs: int = 100) -> float:
        """估算推理時間"""
        self.set_eval_mode()
        dummy_input = torch.randn(input_shape).to(self.device)

        # 暖身運行
        for _ in range(10):
            _ = self(dummy_input)

        # 測量時間
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_time.record()

            for _ in range(num_runs):
                _ = self(dummy_input)

            end_time.record()
            torch.cuda.synchronize()

            avg_time = start_time.elapsed_time(end_time) / num_runs
        else:
            import time
            start = time.time()

            for _ in range(num_runs):
                _ = self(dummy_input)

            end = time.time()
            avg_time = (end - start) * 1000 / num_runs  # 轉換為毫秒

        logger.info(f"平均推理時間: {avg_time:.2f}ms")
        return avg_time


class ModelManager:
    """模型管理器"""

    def __init__(self, models_dir: Union[str, Path] = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: Dict[str, BaseModel] = {}

    def register_model(self, model: BaseModel, model_key: str = None) -> str:
        """註冊模型"""
        key = model_key or f"{model.model_name}_{model.version}"
        self.loaded_models[key] = model
        logger.info(f"已註冊模型: {key}")
        return key

    def get_model(self, model_key: str) -> Optional[BaseModel]:
        """獲取模型"""
        return self.loaded_models.get(model_key)

    def unload_model(self, model_key: str) -> bool:
        """卸載模型"""
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            logger.info(f"已卸載模型: {model_key}")
            return True
        return False

    def list_models(self) -> List[str]:
        """列出所有已載入的模型"""
        return list(self.loaded_models.keys())

    def get_models_info(self) -> Dict[str, Dict[str, Any]]:
        """獲取所有模型的資訊"""
        return {key: model.get_model_info() for key, model in self.loaded_models.items()}

    def save_all_models(self) -> bool:
        """儲存所有模型"""
        try:
            for key, model in self.loaded_models.items():
                save_path = self.models_dir / f"{key}.pth"
                model.save_model(save_path)

            logger.info("所有模型已儲存")
            return True

        except Exception as e:
            logger.error(f"儲存模型失敗: {e}")
            return False


# 全域模型管理器
model_manager = ModelManager()