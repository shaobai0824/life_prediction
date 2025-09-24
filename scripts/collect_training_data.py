#!/usr/bin/env python3
"""
訓練資料收集執行腳本

執行維基百科歷史人物資料收集，生成壽命預測模型的訓練資料
"""

import sys
import asyncio
import logging
from pathlib import Path
import json
from datetime import datetime

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from ml_pipeline.data_collection.wikipedia_collector import WikipediaCollector
from ml_pipeline.data_collection.mock_data_generator import MockDataGenerator

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'logs' / 'data_collection.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


class TrainingDataCollector:
    """訓練資料收集管理器"""

    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = str(project_root / "data" / "collected" / "wikipedia")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化維基百科收集器和模擬資料生成器
        self.wiki_collector = WikipediaCollector(str(self.output_dir))
        self.mock_generator = MockDataGenerator(str(self.output_dir))

        # 收集配置
        self.collection_config = {
            'max_persons': 200,  # 開始先收集200人
            'start_year': 1900,  # 20世紀開始
            'end_year': 2020,    # 到2020年
            'categories': [
                # 政治人物
                "Category:20th-century_American_politicians",
                "Category:Presidents_of_the_United_States",
                "Category:British_politicians",

                # 演藝人員
                "Category:American_film_actors",
                "Category:American_actresses",
                "Category:British_actors",
                "Category:British_actresses",

                # 科學家和學者
                "Category:20th-century_American_scientists",
                "Category:Nobel_Prize_winners",
                "Category:American_mathematicians",

                # 作家和藝術家
                "Category:American_writers",
                "Category:British_writers",
                "Category:American_painters",
                "Category:Musicians"
            ]
        }

    async def run_collection(self) -> dict:
        """執行完整的資料收集流程"""
        try:
            logger.info("Starting training data collection...")
            start_time = datetime.now()

            # 創建日誌目錄
            log_dir = project_root / 'logs'
            log_dir.mkdir(exist_ok=True)

            # 顯示收集配置
            logger.info(f"Collection configuration:")
            logger.info(f"   Target persons: {self.collection_config['max_persons']}")
            logger.info(f"   Time range: {self.collection_config['start_year']}-{self.collection_config['end_year']}")
            logger.info(f"   Categories count: {len(self.collection_config['categories'])}")
            logger.info(f"   Output directory: {self.output_dir}")

            # 嘗試維基百科收集，失敗則使用模擬資料
            logger.info("Starting Wikipedia data collection...")

            result = await self.wiki_collector.collect_historical_data(
                categories=self.collection_config['categories'],
                max_persons=self.collection_config['max_persons'],
                start_year=self.collection_config['start_year'],
                end_year=self.collection_config['end_year']
            )

            # 如果維基百科收集失敗或無資料，使用模擬資料
            if not result['success'] or result.get('training_samples', 0) == 0:
                logger.warning("Wikipedia collection failed or returned no data, using mock data...")
                result = await self.mock_generator.generate_mock_training_data(
                    target_samples=min(50, self.collection_config['max_persons'] * 4)
                )
                result['method'] = 'mock_data'
            else:
                result['method'] = 'wikipedia'

            # 計算總執行時間
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()

            if result['success']:
                logger.info("Data collection completed successfully!")
                logger.info(f"Collection results:")
                logger.info(f"   Persons collected: {result['collected_persons']}")
                logger.info(f"   Training samples: {result['training_samples']}")
                logger.info(f"   Execution time: {total_time:.1f} seconds")
                logger.info(f"   Output location: {result['output_dir']}")

                # 生成收集報告
                await self._generate_collection_report(result, total_time)

                return {
                    'success': True,
                    'message': 'Data collection completed',
                    'result': result,
                    'execution_time': total_time
                }

            else:
                logger.error("Data collection failed")
                logger.error(f"Error: {result.get('error', 'Unknown error')}")

                return {
                    'success': False,
                    'message': 'Data collection failed',
                    'error': result.get('error'),
                    'execution_time': total_time
                }

        except Exception as e:
            logger.error(f"Critical error during execution: {e}")
            return {
                'success': False,
                'message': f'Execution failed: {str(e)}',
                'error': str(e)
            }

    async def _generate_collection_report(self, result: dict, execution_time: float):
        """生成收集報告"""
        try:
            report = {
                'collection_timestamp': datetime.now().isoformat(),
                'execution_time_seconds': execution_time,
                'configuration': self.collection_config,
                'results': result,
                'summary': {
                    'success': result['success'],
                    'total_persons_collected': result.get('collected_persons', 0),
                    'total_training_samples': result.get('training_samples', 0),
                    'collection_stats': result.get('stats', {}),
                    'output_directory': result.get('output_dir', '')
                },
                'recommendations': self._generate_recommendations(result)
            }

            # 保存報告
            report_file = self.output_dir / f"collection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Collection report saved: {report_file}")

        except Exception as e:
            logger.error(f"生成收集報告失敗: {e}")

    def _generate_recommendations(self, result: dict) -> list:
        """生成改進建議"""
        recommendations = []

        stats = result.get('stats', {})

        # 分析收集效率
        if stats.get('total_searched', 0) > 0:
            success_rate = stats.get('valid_persons', 0) / stats.get('total_searched', 1)
            if success_rate < 0.3:
                recommendations.append("建議優化人物篩選條件，提高收集效率")

        # 分析照片數量
        if stats.get('photos_collected', 0) < stats.get('valid_persons', 1) * 2:
            recommendations.append("建議擴大照片來源，增加每人的照片數量")

        # 分析訓練樣本
        training_samples = result.get('training_samples', 0)
        if training_samples < 1000:
            recommendations.append("建議增加收集人數或時間範圍以獲得更多訓練樣本")

        if training_samples >= 1000:
            recommendations.append("訓練樣本數量充足，可以開始模型訓練")

        # 錯誤分析
        if stats.get('errors', 0) > stats.get('total_searched', 0) * 0.1:
            recommendations.append("建議檢查網路連接或API限制，錯誤率較高")

        return recommendations

    async def test_collection(self, test_persons: int = 5):
        """測試資料收集（小樣本）"""
        logger.info(f"Running test collection ({test_persons} persons)...")

        original_max = self.collection_config['max_persons']
        self.collection_config['max_persons'] = test_persons

        try:
            result = await self.run_collection()
            return result
        finally:
            self.collection_config['max_persons'] = original_max


async def main():
    """主執行函數"""
    print("Life Prediction System - Training Data Collector")
    print("=" * 50)

    # 檢查命令行參數
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("Running in test mode...")
        collector = TrainingDataCollector()
        result = await collector.test_collection(test_persons=3)
    else:
        print("Running full collection...")
        collector = TrainingDataCollector()
        result = await collector.run_collection()

    # 顯示結果
    print("\n" + "=" * 50)
    if result['success']:
        print("Collection completed successfully!")
        print(f"Execution time: {result.get('execution_time', 0):.1f} seconds")
        if 'result' in result:
            stats = result['result'].get('stats', {})
            print(f"Collection stats: {stats}")
    else:
        print("Collection failed!")
        print(f"Error: {result.get('message', 'Unknown error')}")

    return result


if __name__ == "__main__":
    try:
        # 運行收集程序
        result = asyncio.run(main())

        # 設定退出碼
        sys.exit(0 if result['success'] else 1)

    except KeyboardInterrupt:
        print("\nUser interrupted execution")
        sys.exit(1)
    except Exception as e:
        print(f"\nProgram execution failed: {e}")
        sys.exit(1)