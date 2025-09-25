#!/usr/bin/env python3
"""
真實圖片壽命預測模型訓練腳本

使用真實的面部圖片特徵進行壽命預測模型訓練
"""

import sys
import os
import logging
from pathlib import Path
import json
import argparse
from datetime import datetime

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from ml_pipeline.training.real_image_trainer import train_with_real_images, RealImageDataset, RealImageLifespanTrainer
from ml_pipeline.models.life_prediction.predictor import LifePredictionModel

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'logs' / 'real_image_training.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


class RealImageTrainingManager:
    """真實圖片模型訓練管理器"""

    def __init__(self):
        self.project_root = project_root
        self.data_dir = self.project_root / "data" / "collected" / "wikipedia"
        self.output_dir = self.project_root / "models" / "trained"
        self.image_cache_dir = self.project_root / "data" / "images" / "cache"
        self.logs_dir = self.project_root / "logs"

        # 創建必要目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def find_latest_dataset(self) -> str:
        """尋找最新的訓練資料集"""
        try:
            pattern = "mock_training_dataset_*.json"
            dataset_files = list(self.data_dir.glob(pattern))

            if not dataset_files:
                raise FileNotFoundError(f"找不到訓練資料檔案，模式: {pattern}")

            # 選擇最新的檔案
            latest_file = max(dataset_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"找到最新的訓練資料檔案: {latest_file}")

            return str(latest_file)

        except Exception as e:
            logger.error(f"尋找訓練資料失敗: {e}")
            raise

    def get_training_config(self, mode: str = "default") -> dict:
        """獲取訓練配置"""
        configs = {
            "quick": {
                'batch_size': 4,  # 較小批量，因為圖片處理較重
                'num_epochs': 20,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'patience': 6,
                'min_delta': 0.001,
                'val_split': 0.2,
                'save_best_model': True,
                'plot_training': True
            },
            "default": {
                'batch_size': 8,
                'num_epochs': 50,
                'learning_rate': 0.0005,
                'weight_decay': 1e-4,
                'patience': 10,
                'min_delta': 0.001,
                'val_split': 0.2,
                'save_best_model': True,
                'plot_training': True
            },
            "comprehensive": {
                'batch_size': 16,
                'num_epochs': 100,
                'learning_rate': 0.0003,
                'weight_decay': 1e-5,
                'patience': 15,
                'min_delta': 0.0005,
                'val_split': 0.15,
                'save_best_model': True,
                'plot_training': True
            }
        }

        return configs.get(mode, configs["default"])

    def analyze_real_image_dataset(self, dataset_file: str):
        """分析真實圖片資料集"""
        try:
            logger.info("正在分析真實圖片訓練資料集...")

            # 創建資料集實例（這會觸發圖片處理）
            dataset = RealImageDataset(
                data_file=dataset_file,
                image_cache_dir=str(self.image_cache_dir)
            )

            # 獲取統計資訊
            stats = dataset.get_statistics()

            if 'error' in stats:
                logger.error(f"資料集分析失敗: {stats['error']}")
                return None

            logger.info("真實圖片資料集分析結果:")
            logger.info(f"  成功處理的樣本數: {stats['num_samples']}")
            logger.info(f"  面部特徵維度: {stats['feature_dim']}")
            logger.info(f"  剩餘壽命統計:")
            label_stats = stats['label_stats']
            logger.info(f"    平均: {label_stats['mean']:.1f} 年")
            logger.info(f"    標準差: {label_stats['std']:.1f} 年")
            logger.info(f"    範圍: {label_stats['min']:.0f} - {label_stats['max']:.0f} 年")

            return dataset, stats

        except Exception as e:
            logger.error(f"真實圖片資料集分析失敗: {e}")
            return None, None

    def train_model(self, mode: str = "default", dataset_file: str = None) -> dict:
        """執行真實圖片模型訓練"""
        try:
            logger.info(f"開始真實圖片模型訓練 (模式: {mode})...")
            start_time = datetime.now()

            # 尋找資料集檔案
            if dataset_file is None:
                dataset_file = self.find_latest_dataset()

            # 分析資料集和處理圖片
            dataset, dataset_stats = self.analyze_real_image_dataset(dataset_file)

            if dataset is None:
                return {
                    'success': False,
                    'error': '資料集準備失敗'
                }

            # 獲取訓練配置
            config = self.get_training_config(mode)
            logger.info(f"訓練配置: {config}")

            # 執行訓練
            result = train_with_real_images(
                data_file=dataset_file,
                output_dir=str(self.output_dir),
                image_cache_dir=str(self.image_cache_dir),
                config=config
            )

            # 計算總時間
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()

            if result['success']:
                logger.info("真實圖片模型訓練成功完成!")
                logger.info(f"訓練時間: {training_time:.1f} 秒")
                logger.info(f"最佳驗證損失: {result['best_val_loss']:.4f}")
                logger.info(f"模型已保存至: {result.get('model_path', 'N/A')}")

                # 添加額外資訊
                result.update({
                    'training_time_seconds': training_time,
                    'dataset_stats': dataset_stats,
                    'dataset_file': dataset_file
                })

                # 生成訓練報告
                self._generate_training_report(result)

            else:
                logger.error("真實圖片模型訓練失敗!")
                logger.error(f"錯誤: {result.get('error', '未知錯誤')}")

            return result

        except Exception as e:
            logger.error(f"真實圖片模型訓練過程發生錯誤: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _generate_training_report(self, training_result: dict):
        """生成訓練報告"""
        try:
            report = {
                'training_timestamp': datetime.now().isoformat(),
                'training_type': 'real_image_features',
                'success': training_result['success'],
                'training_time_seconds': training_result.get('training_time_seconds', 0),
                'dataset_info': {
                    'file': training_result.get('dataset_file'),
                    'stats': training_result.get('dataset_stats', {})
                },
                'model_performance': {
                    'best_val_loss': training_result.get('best_val_loss'),
                    'final_metrics': training_result.get('final_metrics', {})
                },
                'training_config': training_result.get('config', {}),
                'model_path': training_result.get('model_path'),
                'recommendations': self._generate_recommendations(training_result)
            }

            # 保存報告
            report_file = self.output_dir / f"real_image_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"真實圖片訓練報告已保存至: {report_file}")

        except Exception as e:
            logger.error(f"生成訓練報告失敗: {e}")

    def _generate_recommendations(self, training_result: dict) -> list:
        """生成改進建議"""
        recommendations = []

        if training_result.get('success'):
            final_metrics = training_result.get('final_metrics', {})
            mae = final_metrics.get('mae', float('inf'))
            r2 = final_metrics.get('r2', -1)

            # 真實圖片特徵的評估標準
            recommendations.append("✅ 成功使用真實面部特徵進行訓練")

            if mae < 10:
                recommendations.append("🎯 模型精度優秀，真實面部特徵效果良好")
            elif mae < 15:
                recommendations.append("👍 模型精度良好，建議增加更多訓練樣本")
            elif mae < 20:
                recommendations.append("⚠️ 模型精度中等，建議優化面部特徵提取或增加資料")
            else:
                recommendations.append("❌ 模型精度較低，建議檢查面部檢測和特徵提取流程")

            # 資料量分析
            dataset_stats = training_result.get('dataset_stats', {})
            num_samples = dataset_stats.get('num_samples', 0)

            if num_samples < 20:
                recommendations.append("📈 訓練樣本較少，建議收集更多高品質面部圖片")
            elif num_samples < 50:
                recommendations.append("📊 訓練樣本中等，建議持續增加多樣化的面部資料")
            else:
                recommendations.append("✅ 訓練樣本充足，可以考慮優化模型架構")

            recommendations.append("🔬 建議與模擬特徵的結果對比，驗證真實特徵的有效性")

        else:
            recommendations.append("❌ 訓練失敗，請檢查面部檢測器和特徵提取器的配置")
            recommendations.append("🔍 建議檢查圖片品質和面部檢測的成功率")

        return recommendations

    def test_real_image_model(self, model_path: str, test_samples: int = 3):
        """測試真實圖片訓練的模型"""
        try:
            logger.info(f"正在測試真實圖片模型: {model_path}")

            # 載入模型
            from ml_pipeline.training.trainer import load_model
            model, config = load_model(model_path)

            # 載入測試資料
            dataset_file = self.find_latest_dataset()
            dataset = RealImageDataset(
                data_file=dataset_file,
                image_cache_dir=str(self.image_cache_dir)
            )

            # 執行測試預測
            model.eval()
            test_samples = min(test_samples, len(dataset))

            logger.info(f"測試 {test_samples} 個真實圖片樣本:")

            for i in range(test_samples):
                features, label = dataset[i]
                features = features.unsqueeze(0)  # 添加批量維度

                # 預測
                prediction_result = model.predict_lifespan(features)

                actual = float(label[0])
                predicted = prediction_result['predicted_lifespan']
                error = abs(actual - predicted)

                logger.info(f"  樣本 {i+1} (真實面部特徵):")
                logger.info(f"    實際剩餘年數: {actual:.1f}")
                logger.info(f"    預測剩餘年數: {predicted:.1f}")
                logger.info(f"    誤差: {error:.1f} 年")
                logger.info(f"    信心度: {prediction_result['confidence']:.2f}")

            logger.info("真實圖片模型測試完成!")

        except Exception as e:
            logger.error(f"真實圖片模型測試失敗: {e}")


def main():
    """主執行函數"""
    parser = argparse.ArgumentParser(description='真實圖片壽命預測模型訓練腳本')
    parser.add_argument('--mode', choices=['quick', 'default', 'comprehensive'],
                       default='quick', help='訓練模式')
    parser.add_argument('--dataset', type=str, help='指定訓練資料檔案路徑')
    parser.add_argument('--test', type=str, help='測試已訓練的真實圖片模型檔案')
    parser.add_argument('--analyze-only', action='store_true', help='僅分析資料集，不進行訓練')

    args = parser.parse_args()

    print("壽命預測系統 - 真實圖片模型訓練")
    print("=" * 50)

    try:
        manager = RealImageTrainingManager()

        if args.test:
            # 測試模式
            manager.test_real_image_model(args.test)
        elif args.analyze_only:
            # 僅分析模式
            dataset_file = args.dataset or manager.find_latest_dataset()
            dataset, stats = manager.analyze_real_image_dataset(dataset_file)
            if stats:
                print("\n真實圖片特徵提取成功!")
                print(f"處理樣本數: {stats['num_samples']}")
                print(f"特徵維度: {stats['feature_dim']}")
        else:
            # 訓練模式
            result = manager.train_model(mode=args.mode, dataset_file=args.dataset)

            print("=" * 50)
            if result['success']:
                print("🎉 真實圖片模型訓練成功完成!")
                print(f"訓練時間: {result.get('training_time_seconds', 0):.1f} 秒")
                metrics = result.get('final_metrics', {})
                if metrics:
                    print(f"模型性能指標 (真實面部特徵):")
                    print(f"  MAE: {metrics.get('mae', 0):.2f} 年")
                    print(f"  RMSE: {metrics.get('rmse', 0):.2f} 年")
                    print(f"  R²: {metrics.get('r2', 0):.3f}")
                print(f"模型已保存至: {result.get('model_path', 'N/A')}")
            else:
                print("❌ 真實圖片模型訓練失敗!")
                print(f"錯誤: {result.get('error', '未知錯誤')}")
                return 1

    except KeyboardInterrupt:
        print("\n用戶中斷執行")
        return 1
    except Exception as e:
        print(f"\n程序執行失敗: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())