"""
壽命預測模型

基於面部特徵預測個體壽命的深度學習模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Union
import logging
import json
from datetime import datetime

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class LifePredictionModel(BaseModel):
    """壽命預測模型"""

    def __init__(self, input_dim: int = 512, hidden_dims: List[int] = None, device: str = None):
        super().__init__("LifePredictionModel", "1.0.0")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [256, 128, 64, 32]

        if device:
            self.device = torch.device(device)

        # 建立預測網路
        self._build_network()
        self.to(self.device)

        # 預測範圍設定
        self.min_age = 0
        self.max_age = 120
        self.confidence_threshold = 0.7

    def _build_network(self):
        """建立壽命預測網路"""
        try:
            # 主要預測網路
            layers = []
            in_dim = self.input_dim

            for hidden_dim in self.hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3)
                ])
                in_dim = hidden_dim

            # 輸出層
            layers.append(nn.Linear(in_dim, 1))
            self.prediction_network = nn.Sequential(*layers)

            # 不確定性估計網路
            self.uncertainty_network = nn.Sequential(
                nn.Linear(self.input_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Softplus()  # 確保輸出為正值
            )

            # 輔助分析網路
            self.analysis_networks = nn.ModuleDict({
                'health_risk': nn.Sequential(
                    nn.Linear(self.input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 5),  # 5個風險等級
                    nn.Softmax(dim=1)
                ),
                'lifestyle_factors': nn.Sequential(
                    nn.Linear(self.input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 4),  # 4個生活方式因子
                    nn.Sigmoid()
                ),
                'genetic_factors': nn.Sequential(
                    nn.Linear(self.input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 3),  # 3個遺傳因子
                    nn.Sigmoid()
                )
            })

            # 初始化權重
            self._initialize_weights()

            logger.info("壽命預測網路建立成功")

        except Exception as e:
            logger.error(f"網路建立失敗: {e}")
            raise

    def _initialize_weights(self):
        """初始化權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向傳播"""
        # 主要預測
        life_prediction = self.prediction_network(x)

        # 不確定性估計
        uncertainty = self.uncertainty_network(x)

        # 輔助分析
        analyses = {}
        for name, network in self.analysis_networks.items():
            analyses[name] = network(x)

        return {
            'life_prediction': life_prediction,
            'uncertainty': uncertainty,
            'analyses': analyses
        }

    def preprocess(self, features: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """預處理特徵"""
        try:
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float()

            # 確保正確的形狀
            if features.dim() == 1:
                features = features.unsqueeze(0)

            # 檢查特徵維度
            if features.shape[-1] != self.input_dim:
                raise ValueError(f"特徵維度不匹配: 期望 {self.input_dim}, 得到 {features.shape[-1]}")

            # 正規化特徵
            features = F.normalize(features, p=2, dim=1)

            return features.to(self.device)

        except Exception as e:
            logger.error(f"特徵預處理失敗: {e}")
            raise

    def predict_lifespan(self, features: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """預測壽命"""
        try:
            self.set_eval_mode()

            # 預處理特徵
            input_features = self.preprocess(features)

            with torch.no_grad():
                # 前向傳播
                outputs = self.forward(input_features)

                # 處理預測結果
                life_pred = outputs['life_prediction'].cpu().numpy()
                uncertainty = outputs['uncertainty'].cpu().numpy()

                # 確保預測在合理範圍內
                life_pred = np.clip(life_pred, self.min_age, self.max_age)

                # 計算信心度
                confidence = self._calculate_confidence(uncertainty)

                # 處理輔助分析
                analyses = {}
                for name, analysis in outputs['analyses'].items():
                    analyses[name] = analysis.cpu().numpy()

                return {
                    'predicted_lifespan': float(life_pred[0, 0]),
                    'uncertainty': float(uncertainty[0, 0]),
                    'confidence': float(confidence[0]),
                    'prediction_range': {
                        'min': float(life_pred[0, 0] - uncertainty[0, 0]),
                        'max': float(life_pred[0, 0] + uncertainty[0, 0])
                    },
                    'analyses': self._interpret_analyses(analyses)
                }

        except Exception as e:
            logger.error(f"壽命預測失敗: {e}")
            return {}

    def _calculate_confidence(self, uncertainty: np.ndarray) -> np.ndarray:
        """計算預測信心度"""
        # 將不確定性轉換為信心度 (0-1)
        # 不確定性越小，信心度越高
        max_uncertainty = 20.0  # 假設最大不確定性為20年
        confidence = 1.0 - np.clip(uncertainty / max_uncertainty, 0, 1)
        return confidence

    def _interpret_analyses(self, analyses: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """解釋分析結果"""
        interpretations = {}

        # 健康風險分析
        if 'health_risk' in analyses:
            health_risk = analyses['health_risk'][0]
            risk_levels = ['極低', '低', '中等', '高', '極高']
            risk_index = np.argmax(health_risk)
            interpretations['health_risk'] = {
                'level': risk_levels[risk_index],
                'score': float(health_risk[risk_index]),
                'distribution': health_risk.tolist()
            }

        # 生活方式因子分析
        if 'lifestyle_factors' in analyses:
            lifestyle = analyses['lifestyle_factors'][0]
            factors = ['運動習慣', '飲食質量', '壓力管理', '睡眠質量']
            interpretations['lifestyle_factors'] = {
                factor: {
                    'score': float(lifestyle[i]),
                    'level': self._score_to_level(lifestyle[i])
                }
                for i, factor in enumerate(factors)
            }

        # 遺傳因子分析
        if 'genetic_factors' in analyses:
            genetic = analyses['genetic_factors'][0]
            factors = ['長壽基因', '疾病易感性', '代謝效率']
            interpretations['genetic_factors'] = {
                factor: {
                    'score': float(genetic[i]),
                    'level': self._score_to_level(genetic[i])
                }
                for i, factor in enumerate(factors)
            }

        return interpretations

    def _score_to_level(self, score: float) -> str:
        """將數值分數轉換為等級"""
        if score >= 0.8:
            return '優秀'
        elif score >= 0.6:
            return '良好'
        elif score >= 0.4:
            return '中等'
        elif score >= 0.2:
            return '較差'
        else:
            return '不佳'

    def predict(self, input_data: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """進行完整的壽命預測"""
        try:
            # 預測壽命
            prediction_result = self.predict_lifespan(input_data)

            if not prediction_result:
                return {
                    'success': False,
                    'message': '壽命預測失敗',
                    'prediction': None
                }

            # 生成預測報告
            report = self._generate_prediction_report(prediction_result)

            return {
                'success': True,
                'message': '壽命預測完成',
                'prediction': prediction_result,
                'report': report,
                'timestamp': datetime.now().isoformat(),
                'model_version': self.version
            }

        except Exception as e:
            logger.error(f"完整預測失敗: {e}")
            return {
                'success': False,
                'message': f'預測失敗: {str(e)}',
                'prediction': None
            }

    def _generate_prediction_report(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """生成預測報告"""
        try:
            lifespan = prediction['predicted_lifespan']
            confidence = prediction['confidence']
            analyses = prediction.get('analyses', {})

            # 總體評估
            overall_assessment = "基於面部特徵分析，"
            if confidence >= 0.8:
                overall_assessment += "預測結果具有較高可信度。"
            elif confidence >= 0.6:
                overall_assessment += "預測結果具有中等可信度。"
            else:
                overall_assessment += "預測結果可信度較低，建議參考其他健康指標。"

            # 健康建議
            health_suggestions = []
            if 'health_risk' in analyses:
                risk_level = analyses['health_risk']['level']
                if risk_level in ['高', '極高']:
                    health_suggestions.append("建議定期進行健康檢查")
                    health_suggestions.append("注意疾病預防和早期篩檢")

            if 'lifestyle_factors' in analyses:
                lifestyle = analyses['lifestyle_factors']
                for factor, data in lifestyle.items():
                    if data['score'] < 0.5:
                        if '運動' in factor:
                            health_suggestions.append("增加規律運動習慣")
                        elif '飲食' in factor:
                            health_suggestions.append("改善飲食結構，注重營養均衡")
                        elif '睡眠' in factor:
                            health_suggestions.append("改善睡眠質量，保持規律作息")

            return {
                'predicted_lifespan': lifespan,
                'confidence_level': confidence,
                'assessment': overall_assessment,
                'health_suggestions': health_suggestions,
                'detailed_analysis': analyses,
                'disclaimer': "此預測結果僅供參考，不構成醫療建議。實際壽命受多種因素影響，建議諮詢專業醫療人員。"
            }

        except Exception as e:
            logger.error(f"生成報告失敗: {e}")
            return {'error': '報告生成失敗'}

    def postprocess(self, model_output: torch.Tensor) -> Dict[str, Any]:
        """後處理模型輸出"""
        try:
            if isinstance(model_output, dict):
                processed = {}
                for key, value in model_output.items():
                    processed[key] = value.cpu().numpy()
                return processed
            else:
                return {'output': model_output.cpu().numpy()}

        except Exception as e:
            logger.error(f"後處理失敗: {e}")
            return {}

    def update_prediction_parameters(self, min_age: int = None, max_age: int = None,
                                   confidence_threshold: float = None):
        """更新預測參數"""
        if min_age is not None:
            self.min_age = min_age
        if max_age is not None:
            self.max_age = max_age
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold

        logger.info(f"預測參數已更新: 年齡範圍[{self.min_age}, {self.max_age}], 信心度閾值: {self.confidence_threshold}")


def create_life_predictor(input_dim: int = 512, hidden_dims: List[int] = None, device: str = None) -> LifePredictionModel:
    """建立壽命預測模型"""
    return LifePredictionModel(input_dim=input_dim, hidden_dims=hidden_dims, device=device)