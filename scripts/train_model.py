#!/usr/bin/env python3
"""
模型訓練執行腳本

執行壽命預測模型的訓練流程，使用收集的Wikipedia歷史人物資料
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

from ml_pipeline.training.trainer import train_lifespan_model, LifespanDataset, LifePredictionModel, LifespanTrainer
from ml_pipeline.data_processing.image_processor import ImageProcessor

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'logs' / 'model_training.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


class ModelTrainingManager:
    """模型訓練管理器"""

    def __init__(self):
        self.project_root = project_root
        self.data_dir = self.project_root / "data" / "collected" / "wikipedia"
        self.output_dir = self.project_root / "models" / "trained"
        self.logs_dir = self.project_root / "logs"

        # 創建必要目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)
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
                'batch_size': 8,
                'num_epochs': 30,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'patience': 8,
                'min_delta': 0.001,
                'val_split': 0.2,
                'save_best_model': True,
                'plot_training': True
            },
            "default": {
                'batch_size': 16,
                'num_epochs': 100,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'patience': 15,
                'min_delta': 0.001,
                'val_split': 0.2,
                'save_best_model': True,
                'plot_training': True
            },
            "comprehensive": {
                'batch_size': 32,
                'num_epochs': 200,
                'learning_rate': 0.0005,
                'weight_decay': 1e-5,
                'patience': 25,
                'min_delta': 0.0005,
                'val_split': 0.15,
                'save_best_model': True,
                'plot_training': True
            }
        }

        return configs.get(mode, configs["default"])

    def analyze_dataset(self, dataset_file: str):
        """分析資料集"""
        try:
            logger.info("正在分析訓練資料集...")

            with open(dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 基本統計
            num_samples = len(data)

            # 分析各個維度
            ages = [sample['age_at_photo'] for sample in data]
            lifespans = [sample['total_lifespan'] for sample in data]
            remaining_years = [sample['remaining_lifespan_years'] for sample in data]

            # 統計資訊
            stats = {
                'total_samples': num_samples,
                'age_stats': {
                    'min': min(ages),
                    'max': max(ages),
                    'mean': sum(ages) / len(ages)
                },
                'lifespan_stats': {
                    'min': min(lifespans),
                    'max': max(lifespans),
                    'mean': sum(lifespans) / len(lifespans)
                },
                'remaining_years_stats': {
                    'min': min(remaining_years),
                    'max': max(remaining_years),
                    'mean': sum(remaining_years) / len(remaining_years)
                },
                'persons': list(set(sample['person_name'] for sample in data)),
                'nationalities': list(set(sample['nationality'] for sample in data)),
                'occupations': list(set(sample['occupation'] for sample in data))
            }

            logger.info("資料集分析結果:")
            logger.info(f"  總樣本數: {stats['total_samples']}")
            logger.info(f"  人物數量: {len(stats['persons'])}")
            logger.info(f"  年齡範圍: {stats['age_stats']['min']}-{stats['age_stats']['max']} (平均: {stats['age_stats']['mean']:.1f})")
            logger.info(f"  壽命範圍: {stats['lifespan_stats']['min']}-{stats['lifespan_stats']['max']} (平均: {stats['lifespan_stats']['mean']:.1f})")
            logger.info(f"  剩餘年數範圍: {stats['remaining_years_stats']['min']}-{stats['remaining_years_stats']['max']} (平均: {stats['remaining_years_stats']['mean']:.1f})")
            logger.info(f"  國籍: {', '.join(stats['nationalities'])}")
            logger.info(f"  職業: {', '.join(stats['occupations'])}")

            return stats

        except Exception as e:
            logger.error(f"資料集分析失敗: {e}")
            raise

    def train_model(self, mode: str = "default", dataset_file: str = None) -> dict:
        """執行模型訓練"""
        try:
            logger.info(f"開始模型訓練 (模式: {mode})...")
            start_time = datetime.now()

            # 尋找資料集檔案
            if dataset_file is None:
                dataset_file = self.find_latest_dataset()

            # 分析資料集
            dataset_stats = self.analyze_dataset(dataset_file)

            # 獲取訓練配置
            config = self.get_training_config(mode)
            logger.info(f"訓練配置: {config}")

            # 執行訓練
            result = train_lifespan_model(
                data_file=dataset_file,
                output_dir=str(self.output_dir),
                config=config
            )

            # 計算總時間
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()

            if result['success']:
                logger.info("模型訓練成功完成!")
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
                logger.error("模型訓練失敗!")
                logger.error(f"錯誤: {result.get('error', '未知錯誤')}")

            return result

        except Exception as e:
            logger.error(f"模型訓練過程發生錯誤: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _generate_training_report(self, training_result: dict):
        """生成訓練報告"""
        try:
            report = {
                'training_timestamp': datetime.now().isoformat(),
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
            report_file = self.output_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"訓練報告已保存至: {report_file}")

        except Exception as e:
            logger.error(f"生成訓練報告失敗: {e}")

    def _generate_recommendations(self, training_result: dict) -> list:
        """生成改進建議"""
        recommendations = []

        # 分析訓練結果
        if training_result.get('success'):
            final_metrics = training_result.get('final_metrics', {})
            mae = final_metrics.get('mae', float('inf'))
            r2 = final_metrics.get('r2', -1)

            # MAE分析
            if mae < 5:
                recommendations.append("模型精度優秀，平均誤差小於5年")
            elif mae < 10:
                recommendations.append("模型精度良好，平均誤差小於10年")
            elif mae < 15:
                recommendations.append("模型精度中等，建議收集更多訓練資料或調整網路結構")
            else:
                recommendations.append("模型精度較低，建議重新檢查資料品質和模型設計")

            # R²分析
            if r2 > 0.8:
                recommendations.append("模型擬合度優秀，可以進行生產部署")
            elif r2 > 0.6:
                recommendations.append("模型擬合度良好，建議進一步優化")
            elif r2 > 0.4:
                recommendations.append("模型擬合度中等，需要改進特徵工程或增加資料")
            else:
                recommendations.append("模型擬合度較低，建議重新設計模型架構")

            # 資料量分析
            dataset_stats = training_result.get('dataset_stats', {})
            total_samples = dataset_stats.get('total_samples', 0)

            if total_samples < 50:
                recommendations.append("訓練樣本較少，建議收集更多歷史人物資料")
            elif total_samples < 100:
                recommendations.append("訓練樣本中等，建議適度增加資料量")
            else:
                recommendations.append("訓練樣本充足，可以考慮增加模型複雜度")

        else:
            recommendations.append("訓練失敗，請檢查資料格式和系統配置")

        return recommendations

    def test_model(self, model_path: str, test_samples: int = 5):
        """測試已訓練的模型"""
        try:
            logger.info(f"正在測試模型: {model_path}")

            # 載入模型
            from ml_pipeline.training.trainer import load_model
            model, config = load_model(model_path)

            # 載入測試資料
            dataset_file = self.find_latest_dataset()
            dataset = LifespanDataset(dataset_file)

            # 執行測試預測
            model.eval()
            test_data = dataset.data[:test_samples]

            logger.info(f"測試 {len(test_data)} 個樣本:")

            for i, sample in enumerate(test_data):
                # 提取特徵
                feature = dataset._create_mock_features(sample)
                feature_normalized = dataset.scaler.transform([feature])

                # 預測
                prediction_result = model.predict_lifespan(feature_normalized)

                actual = sample['remaining_lifespan_years']
                predicted = prediction_result['predicted_lifespan']
                error = abs(actual - predicted)

                logger.info(f"  樣本 {i+1}: {sample['person_name']} (年齡 {sample['age_at_photo']})")
                logger.info(f"    實際剩餘年數: {actual}")
                logger.info(f"    預測剩餘年數: {predicted:.1f}")
                logger.info(f"    誤差: {error:.1f} 年")
                logger.info(f"    信心度: {prediction_result['confidence']:.2f}")

            logger.info("模型測試完成!")

        except Exception as e:
            logger.error(f"模型測試失敗: {e}")


def main():
    """主執行函數"""
    parser = argparse.ArgumentParser(description='壽命預測模型訓練腳本')
    parser.add_argument('--mode', choices=['quick', 'default', 'comprehensive'],
                       default='default', help='訓練模式')
    parser.add_argument('--dataset', type=str, help='指定訓練資料檔案路徑')
    parser.add_argument('--test', type=str, help='測試已訓練的模型檔案')
    parser.add_argument('--analyze-only', action='store_true', help='僅分析資料集，不進行訓練')

    args = parser.parse_args()

    print("壽命預測系統 - 模型訓練")
    print("=" * 50)

    try:
        manager = ModelTrainingManager()

        if args.test:
            # 測試模式
            manager.test_model(args.test)
        elif args.analyze_only:
            # 僅分析模式
            dataset_file = args.dataset or manager.find_latest_dataset()
            manager.analyze_dataset(dataset_file)
        else:
            # 訓練模式
            result = manager.train_model(mode=args.mode, dataset_file=args.dataset)

            print("=" * 50)
            if result['success']:
                print("訓練成功完成!")
                print(f"訓練時間: {result.get('training_time_seconds', 0):.1f} 秒")
                metrics = result.get('final_metrics', {})
                if metrics:
                    print(f"模型性能指標:")
                    print(f"  MAE: {metrics.get('mae', 0):.2f} 年")
                    print(f"  RMSE: {metrics.get('rmse', 0):.2f} 年")
                    print(f"  R²: {metrics.get('r2', 0):.3f}")
                print(f"模型已保存至: {result.get('model_path', 'N/A')}")
            else:
                print("訓練失敗!")
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