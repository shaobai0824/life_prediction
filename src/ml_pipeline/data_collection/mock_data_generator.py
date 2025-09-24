"""
模擬訓練資料生成器

當維基百科API不可用時，生成模擬的歷史人物資料用於展示系統功能
"""

import json
import random
import asyncio
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class MockDataGenerator:
    """模擬資料生成器"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 模擬歷史人物資料庫
        self.mock_persons = [
            {
                'name': 'Albert Einstein',
                'birth_year': 1879,
                'death_year': 1955,
                'nationality': 'German-American',
                'occupation': 'Physicist',
                'photos': [
                    {'year': 1905, 'age': 26},
                    {'year': 1921, 'age': 42},
                    {'year': 1935, 'age': 56},
                    {'year': 1950, 'age': 71}
                ]
            },
            {
                'name': 'Charlie Chaplin',
                'birth_year': 1889,
                'death_year': 1977,
                'nationality': 'British',
                'occupation': 'Actor',
                'photos': [
                    {'year': 1915, 'age': 26},
                    {'year': 1925, 'age': 36},
                    {'year': 1940, 'age': 51},
                    {'year': 1960, 'age': 71}
                ]
            },
            {
                'name': 'Winston Churchill',
                'birth_year': 1874,
                'death_year': 1965,
                'nationality': 'British',
                'occupation': 'Politician',
                'photos': [
                    {'year': 1900, 'age': 26},
                    {'year': 1920, 'age': 46},
                    {'year': 1940, 'age': 66},
                    {'year': 1960, 'age': 86}
                ]
            },
            {
                'name': 'Ernest Hemingway',
                'birth_year': 1899,
                'death_year': 1961,
                'nationality': 'American',
                'occupation': 'Writer',
                'photos': [
                    {'year': 1925, 'age': 26},
                    {'year': 1935, 'age': 36},
                    {'year': 1950, 'age': 51},
                    {'year': 1960, 'age': 61}
                ]
            },
            {
                'name': 'Marilyn Monroe',
                'birth_year': 1926,
                'death_year': 1962,
                'nationality': 'American',
                'occupation': 'Actress',
                'photos': [
                    {'year': 1946, 'age': 20},
                    {'year': 1950, 'age': 24},
                    {'year': 1955, 'age': 29},
                    {'year': 1961, 'age': 35}
                ]
            },
            {
                'name': 'John F. Kennedy',
                'birth_year': 1917,
                'death_year': 1963,
                'nationality': 'American',
                'occupation': 'Politician',
                'photos': [
                    {'year': 1940, 'age': 23},
                    {'year': 1950, 'age': 33},
                    {'year': 1960, 'age': 43},
                    {'year': 1963, 'age': 46}
                ]
            },
            {
                'name': 'Pablo Picasso',
                'birth_year': 1881,
                'death_year': 1973,
                'nationality': 'Spanish',
                'occupation': 'Artist',
                'photos': [
                    {'year': 1900, 'age': 19},
                    {'year': 1920, 'age': 39},
                    {'year': 1940, 'age': 59},
                    {'year': 1960, 'age': 79},
                    {'year': 1970, 'age': 89}
                ]
            },
            {
                'name': 'Elvis Presley',
                'birth_year': 1935,
                'death_year': 1977,
                'nationality': 'American',
                'occupation': 'Musician',
                'photos': [
                    {'year': 1954, 'age': 19},
                    {'year': 1960, 'age': 25},
                    {'year': 1968, 'age': 33},
                    {'year': 1975, 'age': 40}
                ]
            },
            {
                'name': 'Grace Kelly',
                'birth_year': 1929,
                'death_year': 1982,
                'nationality': 'American',
                'occupation': 'Actress',
                'photos': [
                    {'year': 1950, 'age': 21},
                    {'year': 1955, 'age': 26},
                    {'year': 1965, 'age': 36},
                    {'year': 1980, 'age': 51}
                ]
            },
            {
                'name': 'Frank Sinatra',
                'birth_year': 1915,
                'death_year': 1998,
                'nationality': 'American',
                'occupation': 'Musician',
                'photos': [
                    {'year': 1940, 'age': 25},
                    {'year': 1950, 'age': 35},
                    {'year': 1970, 'age': 55},
                    {'year': 1990, 'age': 75}
                ]
            }
        ]

    async def generate_mock_training_data(self, target_samples: int = 100) -> Dict[str, Any]:
        """生成模擬訓練資料"""
        try:
            logger.info(f"Generating mock training data with {target_samples} samples...")

            # 統計資訊
            stats = {
                'total_searched': 0,
                'valid_persons': 0,
                'photos_collected': 0,
                'training_samples': 0,
                'errors': 0
            }

            # 生成訓練樣本
            training_samples = []
            collected_persons = []

            for person in self.mock_persons:
                try:
                    stats['total_searched'] += 1

                    # 建立人物記錄
                    person_record = {
                        'name': person['name'],
                        'birth_date': date(person['birth_year'], 6, 15).isoformat(),
                        'death_date': date(person['death_year'], 6, 15).isoformat(),
                        'nationality': person['nationality'],
                        'occupation': person['occupation'],
                        'lifespan': person['death_year'] - person['birth_year'],
                        'photos': []
                    }

                    # 為每張照片生成訓練樣本
                    for photo in person['photos']:
                        photo_year = photo['year']
                        age_at_photo = photo['age']
                        remaining_years = person['death_year'] - photo_year

                        if remaining_years >= 0:  # 確保照片在死亡前拍攝
                            # 生成模擬照片資訊
                            mock_photo = {
                                'title': f"File:{person['name'].replace(' ', '_')}_{photo_year}.jpg",
                                'url': f"https://upload.wikimedia.org/wikipedia/commons/mock/{person['name'].replace(' ', '_')}_{photo_year}.jpg",
                                'width': random.randint(300, 800),
                                'height': random.randint(300, 800),
                                'timestamp': f"{photo_year}-06-15T12:00:00Z",
                                'photo_date': date(photo_year, 6, 15).isoformat(),
                                'size': random.randint(50000, 500000)
                            }

                            person_record['photos'].append(mock_photo)
                            stats['photos_collected'] += 1

                            # 創建訓練樣本
                            training_sample = {
                                'person_name': person['name'],
                                'photo_url': mock_photo['url'],
                                'photo_date': mock_photo['photo_date'],
                                'death_date': person_record['death_date'],
                                'birth_date': person_record['birth_date'],
                                'remaining_lifespan_years': remaining_years,
                                'age_at_photo': age_at_photo,
                                'total_lifespan': person_record['lifespan'],
                                'nationality': person['nationality'],
                                'occupation': person['occupation'],
                                'image_width': mock_photo['width'],
                                'image_height': mock_photo['height'],
                                'wikipedia_url': f"https://en.wikipedia.org/wiki/{person['name'].replace(' ', '_')}"
                            }

                            training_samples.append(training_sample)
                            stats['training_samples'] += 1

                            # 如果達到目標樣本數就停止
                            if len(training_samples) >= target_samples:
                                break

                    collected_persons.append(person_record)
                    stats['valid_persons'] += 1

                except Exception as e:
                    logger.warning(f"處理人物 {person['name']} 失敗: {e}")
                    stats['errors'] += 1
                    continue

                # 如果達到目標樣本數就停止
                if len(training_samples) >= target_samples:
                    break

            # 保存資料
            await self._save_mock_data(collected_persons, training_samples, stats)

            logger.info(f"Mock data generation completed: {len(training_samples)} samples")

            return {
                'success': True,
                'collected_persons': len(collected_persons),
                'training_samples': len(training_samples),
                'stats': stats,
                'output_dir': str(self.output_dir)
            }

        except Exception as e:
            logger.error(f"Mock data generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _save_mock_data(
        self,
        persons: List[Dict[str, Any]],
        training_samples: List[Dict[str, Any]],
        stats: Dict[str, Any]
    ):
        """保存模擬資料"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # 保存人物資料
            persons_file = self.output_dir / f"mock_persons_{timestamp}.json"
            with open(persons_file, 'w', encoding='utf-8') as f:
                json.dump(persons, f, indent=2, ensure_ascii=False, default=str)

            # 保存訓練資料集
            dataset_file = self.output_dir / f"mock_training_dataset_{timestamp}.json"
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump(training_samples, f, indent=2, ensure_ascii=False, default=str)

            # 保存統計報告
            stats_file = self.output_dir / f"mock_stats_{timestamp}.json"
            enhanced_stats = {
                **stats,
                'generation_method': 'mock_data_generator',
                'generation_time': timestamp,
                'note': 'This is mock data for demonstration purposes only'
            }
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_stats, f, indent=2, ensure_ascii=False)

            logger.info(f"Mock data saved to: {self.output_dir}")

        except Exception as e:
            logger.error(f"Saving mock data failed: {e}")

    def create_sample_analysis(self) -> Dict[str, Any]:
        """創建樣本分析報告"""
        analysis = {
            'sample_distribution': {
                'by_decade': {},
                'by_nationality': {},
                'by_occupation': {},
                'by_lifespan_range': {}
            },
            'statistics': {
                'average_lifespan': 0,
                'median_lifespan': 0,
                'min_lifespan': float('inf'),
                'max_lifespan': 0,
                'total_samples': len(self.mock_persons)
            }
        }

        lifespans = []

        for person in self.mock_persons:
            lifespan = person['death_year'] - person['birth_year']
            lifespans.append(lifespan)

            # 按年代分組
            decade = f"{person['birth_year']//10*10}s"
            analysis['sample_distribution']['by_decade'][decade] = \
                analysis['sample_distribution']['by_decade'].get(decade, 0) + 1

            # 按國籍分組
            nationality = person['nationality']
            analysis['sample_distribution']['by_nationality'][nationality] = \
                analysis['sample_distribution']['by_nationality'].get(nationality, 0) + 1

            # 按職業分組
            occupation = person['occupation']
            analysis['sample_distribution']['by_occupation'][occupation] = \
                analysis['sample_distribution']['by_occupation'].get(occupation, 0) + 1

            # 按壽命範圍分組
            if lifespan < 50:
                range_key = 'under_50'
            elif lifespan < 70:
                range_key = '50_70'
            elif lifespan < 90:
                range_key = '70_90'
            else:
                range_key = 'over_90'

            analysis['sample_distribution']['by_lifespan_range'][range_key] = \
                analysis['sample_distribution']['by_lifespan_range'].get(range_key, 0) + 1

        # 計算統計數據
        analysis['statistics']['average_lifespan'] = np.mean(lifespans)
        analysis['statistics']['median_lifespan'] = np.median(lifespans)
        analysis['statistics']['min_lifespan'] = min(lifespans)
        analysis['statistics']['max_lifespan'] = max(lifespans)

        return analysis


async def main():
    """測試模擬資料生成"""
    output_dir = "./data/collected/mock"
    generator = MockDataGenerator(output_dir)

    # 生成模擬資料
    result = await generator.generate_mock_training_data(target_samples=50)
    print(json.dumps(result, indent=2))

    # 生成分析報告
    analysis = generator.create_sample_analysis()
    print("\nSample Analysis:")
    print(json.dumps(analysis, indent=2, default=float))


if __name__ == "__main__":
    asyncio.run(main())