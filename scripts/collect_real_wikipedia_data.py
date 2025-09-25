#!/usr/bin/env python3
"""
真實Wikipedia資料收集腳本

修復並執行真實的Wikipedia API呼叫，收集具有真實圖片和壽命資料的訓練資料
"""

import sys
import os
import logging
import asyncio
from pathlib import Path
import json
import argparse
from datetime import datetime

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from ml_pipeline.data_collection.wikipedia_collector import WikipediaCollector

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(project_root / 'logs' / 'real_wikipedia_collection.log', encoding='utf-8')
    ]
)

# 確保stdout使用UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)


class RealWikipediaDataManager:
    """真實Wikipedia資料收集管理器"""

    def __init__(self):
        self.project_root = project_root
        self.output_dir = self.project_root / "data" / "collected" / "wikipedia"
        self.image_cache_dir = self.project_root / "data" / "images" / "cache"
        self.logs_dir = self.project_root / "logs"

        # 創建必要目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def get_famous_deceased_persons(self) -> list:
        """獲取知名已故人物列表（小規模測試用）"""
        return [
            "Albert Einstein",
            "Winston Churchill",
            "Franklin D. Roosevelt",
            "John F. Kennedy",
            "Martin Luther King Jr.",
            "Nelson Mandela",
            "Marilyn Monroe",
            "Elvis Presley",
            "Michael Jackson",
            "Princess Diana",
            "Steve Jobs",
            "Muhammad Ali",
            "Audrey Hepburn",
            "Charlie Chaplin",
            "Pablo Picasso"
        ]

    async def test_wikipedia_api_access(self) -> bool:
        """測試Wikipedia API存取"""
        try:
            import aiohttp

            logger.info("測試Wikipedia API存取...")

            headers = {
                'User-Agent': 'LifePredictionBot/1.0 (https://example.com/contact; research@example.com) Python/aiohttp'
            }
            async with aiohttp.ClientSession(headers=headers) as session:
                # 測試基本API呼叫
                params = {
                    'action': 'query',
                    'format': 'json',
                    'titles': 'Albert Einstein',
                    'prop': 'info'
                }

                # 使用標準的Wikipedia API而不是REST API
                api_url = 'https://en.wikipedia.org/w/api.php'

                async with session.get(api_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info("Wikipedia API存取成功")
                        logger.info(f"測試資料: {list(data.get('query', {}).get('pages', {}).keys())}")
                        return True
                    else:
                        logger.error(f"Wikipedia API存取失敗: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"❌ Wikipedia API測試失敗: {e}")
            return False

    async def collect_small_dataset(self, max_persons: int = 10) -> dict:
        """收集小規模真實資料集"""
        try:
            logger.info(f"開始收集小規模真實Wikipedia資料集 (最多 {max_persons} 人)...")

            # 初始化收集器
            collector = WikipediaCollector()

            # 獲取已故名人列表
            famous_persons = self.get_famous_deceased_persons()[:max_persons]
            logger.info(f"目標收集人物: {famous_persons}")

            # 收集資料
            collected_persons = []

            import aiohttp
            headers = {
                'User-Agent': 'LifePredictionBot/1.0 (https://example.com/contact; research@example.com) Python/aiohttp'
            }
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers=headers
            ) as session:
                for person_name in famous_persons:
                    try:
                        logger.info(f"正在收集: {person_name}")

                        # 手動調用收集器的內部方法
                        person_info = await collector._get_person_info(session, person_name)

                        if person_info and person_info.get('birth_date') and person_info.get('death_date'):
                            logger.info(f"✅ 成功獲取 {person_name} 的基本資料")

                            # 獲取照片
                            photos = await collector._get_person_photos(session, person_name)
                            logger.info(f"  發現 {len(photos)} 張潛在照片")

                            if photos:
                                # 創建人物資料物件
                                from ml_pipeline.data_collection.wikipedia_collector import PersonData
                                person_data = PersonData(
                                    name=person_info['name'],
                                    birth_date=person_info['birth_date'],
                                    death_date=person_info['death_date'],
                                    wikipedia_url=f"https://en.wikipedia.org/wiki/{person_name.replace(' ', '_')}",
                                    photos=photos[:3],  # 限制每人3張照片
                                    nationality=person_info.get('nationality'),
                                    occupation=person_info.get('occupation'),
                                    cause_of_death=person_info.get('cause_of_death')
                                )

                                collected_persons.append(person_data)
                                logger.info(f"✅ {person_name} 收集完成")
                            else:
                                logger.warning(f"⚠️ {person_name} 沒有可用照片")
                        else:
                            logger.warning(f"⚠️ {person_name} 基本資料不完整")

                    except Exception as e:
                        logger.error(f"❌ 收集 {person_name} 失敗: {e}")
                        continue

                    # 避免過快請求
                    await asyncio.sleep(2)

            logger.info(f"成功收集 {len(collected_persons)} 個人物資料")

            # 生成訓練資料集
            training_dataset = []
            for person in collected_persons:
                if person.photos:
                    for photo in person.photos:
                        if photo.get('photo_date'):
                            remaining_years = person.remaining_lifespan_at_photo(photo['photo_date'])
                            if remaining_years is not None and remaining_years > 0:
                                training_sample = {
                                    'person_name': person.name,
                                    'photo_url': photo['url'],
                                    'photo_date': photo['photo_date'].isoformat() if photo['photo_date'] else None,
                                    'death_date': person.death_date.isoformat(),
                                    'birth_date': person.birth_date.isoformat(),
                                    'remaining_lifespan_years': remaining_years,
                                    'age_at_photo': person.age_at_photo(photo['photo_date']),
                                    'total_lifespan': person.lifespan_years,
                                    'nationality': person.nationality,
                                    'occupation': person.occupation,
                                    'image_width': photo.get('width'),
                                    'image_height': photo.get('height'),
                                    'wikipedia_url': person.wikipedia_url
                                }
                                training_dataset.append(training_sample)

            # 保存結果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # 保存人物資料
            persons_file = self.output_dir / f"real_persons_{timestamp}.json"
            persons_data = [person.to_dict() for person in collected_persons]
            with open(persons_file, 'w', encoding='utf-8') as f:
                json.dump(persons_data, f, indent=2, ensure_ascii=False, default=str)

            # 保存訓練資料集
            dataset_file = self.output_dir / f"real_training_dataset_{timestamp}.json"
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump(training_dataset, f, indent=2, ensure_ascii=False, default=str)

            # 生成統計報告
            stats = {
                'collection_timestamp': timestamp,
                'persons_collected': len(collected_persons),
                'training_samples': len(training_dataset),
                'average_photos_per_person': len(training_dataset) / max(len(collected_persons), 1),
                'persons_with_photos': len([p for p in collected_persons if p.photos]),
                'files_created': {
                    'persons_file': str(persons_file),
                    'dataset_file': str(dataset_file)
                }
            }

            stats_file = self.output_dir / f"real_collection_stats_{timestamp}.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

            logger.info("✅ 真實資料收集完成!")
            logger.info(f"收集人物: {stats['persons_collected']}")
            logger.info(f"訓練樣本: {stats['training_samples']}")
            logger.info(f"資料檔案: {dataset_file}")

            return {
                'success': True,
                'stats': stats,
                'dataset_file': str(dataset_file),
                'persons_file': str(persons_file)
            }

        except Exception as e:
            logger.error(f"真實資料收集失敗: {e}")
            return {
                'success': False,
                'error': str(e)
            }


async def main():
    """主執行函數"""
    parser = argparse.ArgumentParser(description='真實Wikipedia資料收集腳本')
    parser.add_argument('--max-persons', type=int, default=10,
                       help='最多收集人數（預設10人）')
    parser.add_argument('--test-api', action='store_true',
                       help='僅測試Wikipedia API存取')

    args = parser.parse_args()

    print("壽命預測系統 - 真實Wikipedia資料收集")
    print("=" * 50)

    try:
        manager = RealWikipediaDataManager()

        if args.test_api:
            # 測試API存取
            success = await manager.test_wikipedia_api_access()
            if success:
                print("✅ Wikipedia API測試成功")
                return 0
            else:
                print("❌ Wikipedia API測試失敗")
                return 1
        else:
            # 收集真實資料
            result = await manager.collect_small_dataset(max_persons=args.max_persons)

            print("=" * 50)
            if result['success']:
                print("🎉 真實Wikipedia資料收集成功!")
                stats = result['stats']
                print(f"收集人物: {stats['persons_collected']}")
                print(f"訓練樣本: {stats['training_samples']}")
                print(f"資料檔案: {result['dataset_file']}")
            else:
                print("❌ 真實Wikipedia資料收集失敗!")
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
    sys.exit(asyncio.run(main()))