#!/usr/bin/env python3
"""
大量收集已故人物資料腳本

使用Wikipedia的分類和搜索API，自動發現並收集符合條件的已故人物資料：
- 不需要預先指定人名
- 按死亡年份範圍搜索
- 自動過濾條件（有照片、有生卒年月、圖片品質等）
- 大量並行處理
"""

import sys
import os
import logging
import asyncio
import aiohttp
from pathlib import Path
import json
import argparse
from datetime import datetime, date
from typing import Dict, List, Any, Set, Optional
import random

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from ml_pipeline.data_collection.wikipedia_collector import WikipediaCollector
from ml_pipeline.data_collection.image_downloader import RealImageDownloader

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(project_root / 'logs' / 'mass_collection.log', encoding='utf-8')
    ]
)

# 確保stdout使用UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)


class MassDeceasedCollector:
    """大量已故人物收集器"""

    def __init__(self):
        self.project_root = project_root
        self.output_dir = self.project_root / "data" / "collected" / "mass_search"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.collector = WikipediaCollector()
        self.image_downloader = RealImageDownloader()

        self.wiki_api_url = "https://en.wikipedia.org/w/api.php"

        # 統計資料
        self.stats = {
            'pages_searched': 0,
            'deceased_found': 0,
            'with_photos': 0,
            'successfully_collected': 0,
            'errors': 0
        }

    def get_death_year_categories(self, start_year: int, end_year: int) -> List[str]:
        """生成按死亡年份的Wikipedia分類名稱"""
        categories = []
        for year in range(start_year, end_year + 1):
            categories.extend([
                f"{year} deaths",
                f"Deaths in {year}",  # 某些分類使用這種格式
            ])
        return categories

    def get_additional_death_categories(self) -> List[str]:
        """獲取其他死亡相關的分類"""
        return [
            "20th-century deaths",
            "21st-century deaths",
            "People who died in the 1900s",
            "People who died in the 2000s",
            "Deceased people",
            "Deaths by year",
        ]

    async def search_category_members(
        self,
        session: aiohttp.ClientSession,
        category: str,
        limit: int = 100
    ) -> List[str]:
        """搜索分類中的成員"""
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'categorymembers',
                'cmtitle': f'Category:{category}',
                'cmlimit': limit,
                'cmtype': 'page',
                'cmnamespace': 0  # 只要文章命名空間
            }

            async with session.get(self.wiki_api_url, params=params) as response:
                if response.status != 200:
                    return []

                data = await response.json()

                if 'query' not in data or 'categorymembers' not in data['query']:
                    return []

                members = []
                for member in data['query']['categorymembers']:
                    title = member.get('title', '').strip()
                    if title and ':' not in title:  # 排除重定向和其他命名空間
                        members.append(title)

                logger.debug(f"分類 '{category}' 找到 {len(members)} 個成員")
                return members

        except Exception as e:
            logger.error(f"搜索分類 '{category}' 失敗: {e}")
            return []

    async def search_by_death_keyword(
        self,
        session: aiohttp.ClientSession,
        search_terms: List[str],
        limit: int = 100
    ) -> List[str]:
        """使用關鍵字搜索已故人物"""
        found_titles = set()

        for term in search_terms:
            try:
                params = {
                    'action': 'query',
                    'format': 'json',
                    'list': 'search',
                    'srsearch': term,
                    'srlimit': limit,
                    'srnamespace': 0,
                    'srprop': 'snippet'
                }

                async with session.get(self.wiki_api_url, params=params) as response:
                    if response.status != 200:
                        continue

                    data = await response.json()

                    if 'query' not in data or 'search' not in data['query']:
                        continue

                    for result in data['query']['search']:
                        title = result.get('title', '').strip()
                        snippet = result.get('snippet', '').lower()

                        # 檢查是否含有死亡相關詞彙
                        death_indicators = ['died', 'death', 'deceased', 'passed away', 'd.']
                        if title and any(indicator in snippet for indicator in death_indicators):
                            found_titles.add(title)

                logger.debug(f"關鍵字 '{term}' 找到 {len(found_titles)} 個結果")

            except Exception as e:
                logger.error(f"關鍵字搜索 '{term}' 失敗: {e}")

        return list(found_titles)

    async def discover_deceased_persons(
        self,
        session: aiohttp.ClientSession,
        start_year: int = 2000,
        end_year: int = 2023,
        max_persons: int = 100
    ) -> List[str]:
        """發現已故人物"""
        logger.info(f"開始搜索 {start_year}-{end_year} 年間過世的人物...")

        all_candidates = set()

        # 1. 按年份分類搜索
        year_categories = self.get_death_year_categories(start_year, end_year)

        for category in year_categories:
            members = await self.search_category_members(session, category, limit=50)
            all_candidates.update(members)
            self.stats['pages_searched'] += len(members)

            if len(all_candidates) >= max_persons * 3:  # 收集更多候選者以便過濾
                break

            await asyncio.sleep(1)  # 避免過快請求

        # 2. 使用關鍵字搜索補充
        if len(all_candidates) < max_persons * 2:
            search_terms = [
                f"died {year}" for year in range(start_year, end_year + 1, 5)
            ] + [
                "biography death",
                "obituary",
                "died aged",
                "passed away"
            ]

            keyword_results = await self.search_by_death_keyword(session, search_terms[:5], limit=30)
            all_candidates.update(keyword_results)

        logger.info(f"發現 {len(all_candidates)} 個候選人物")

        # 隨機化順序，避免總是收集相同的人物
        candidates_list = list(all_candidates)
        random.shuffle(candidates_list)

        return candidates_list[:max_persons * 2]  # 返回兩倍數量以便後續過濾

    async def is_valid_deceased_person(
        self,
        session: aiohttp.ClientSession,
        title: str
    ) -> Dict[str, Any]:
        """檢查是否為有效的已故人物"""
        try:
            person_info = await self.collector._get_person_info(session, title)

            if not person_info:
                return {'valid': False, 'reason': '無法獲取基本資訊'}

            # 必須有生卒日期
            if not person_info.get('birth_date') or not person_info.get('death_date'):
                return {'valid': False, 'reason': '缺少生卒日期'}

            # 檢查是否有照片
            photos = await self.collector._get_person_photos(session, title)
            if len(photos) < 1:
                return {'valid': False, 'reason': '沒有照片'}

            # 計算年齡是否合理
            birth_date = person_info['birth_date']
            death_date = person_info['death_date']

            if not isinstance(birth_date, date) or not isinstance(death_date, date):
                return {'valid': False, 'reason': '日期格式不正確'}

            age = (death_date - birth_date).days // 365
            if age < 10 or age > 120:
                return {'valid': False, 'reason': f'年齡不合理: {age}歲'}

            return {
                'valid': True,
                'person_info': person_info,
                'photos': photos,
                'age': age
            }

        except Exception as e:
            return {'valid': False, 'reason': f'驗證失敗: {str(e)}'}

    async def collect_batch_persons(
        self,
        session: aiohttp.ClientSession,
        candidates: List[str],
        max_valid: int = 50
    ) -> List[Dict[str, Any]]:
        """批次收集有效人物"""
        logger.info(f"開始批次驗證 {len(candidates)} 個候選人物...")

        valid_persons = []
        processed = 0

        for title in candidates:
            if len(valid_persons) >= max_valid:
                break

            try:
                processed += 1
                logger.info(f"[{processed}/{len(candidates)}] 驗證: {title}")

                validation = await self.is_valid_deceased_person(session, title)

                if validation['valid']:
                    logger.info(f"✅ {title} - 有效 (年齡: {validation['age']}歲, 照片: {len(validation['photos'])}張)")

                    # 生成訓練資料格式
                    person_data = {
                        'person_name': title,
                        'birth_date': validation['person_info']['birth_date'].isoformat(),
                        'death_date': validation['person_info']['death_date'].isoformat(),
                        'age_at_death': validation['age'],
                        'nationality': validation['person_info'].get('nationality'),
                        'occupation': validation['person_info'].get('occupation'),
                        'photos': validation['photos']
                    }

                    valid_persons.append(person_data)
                    self.stats['successfully_collected'] += 1

                else:
                    logger.debug(f"❌ {title} - {validation['reason']}")

                self.stats['deceased_found'] += 1

                # 避免過快請求
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"處理 {title} 時發生錯誤: {e}")
                self.stats['errors'] += 1

        logger.info(f"批次驗證完成，獲得 {len(valid_persons)} 個有效人物")
        return valid_persons

    async def run_mass_collection(
        self,
        start_year: int = 2000,
        end_year: int = 2023,
        max_persons: int = 50,
        output_prefix: str = "mass_collected"
    ) -> Dict[str, Any]:
        """執行大量收集"""
        try:
            logger.info("開始大量已故人物收集...")
            start_time = datetime.now()

            headers = {
                'User-Agent': 'LifePredictionBot/1.0 (https://example.com/contact; research@example.com) Python/aiohttp'
            }

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers=headers
            ) as session:

                # 1. 發現候選人物
                candidates = await self.discover_deceased_persons(
                    session, start_year, end_year, max_persons
                )

                if not candidates:
                    return {
                        'success': False,
                        'error': '未找到任何候選人物'
                    }

                # 2. 批次收集有效人物
                valid_persons = await self.collect_batch_persons(
                    session, candidates, max_persons
                )

                # 3. 保存結果
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = self.output_dir / f"{output_prefix}_{timestamp}.json"

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(valid_persons, f, indent=2, ensure_ascii=False, default=str)

                # 4. 生成統計報告
                end_time = datetime.now()
                collection_time = (end_time - start_time).total_seconds()

                stats_report = {
                    'collection_timestamp': timestamp,
                    'search_parameters': {
                        'start_year': start_year,
                        'end_year': end_year,
                        'max_persons': max_persons
                    },
                    'collection_stats': self.stats,
                    'results': {
                        'valid_persons_found': len(valid_persons),
                        'collection_time_seconds': collection_time,
                        'output_file': str(output_file)
                    }
                }

                stats_file = self.output_dir / f"{output_prefix}_stats_{timestamp}.json"
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(stats_report, f, indent=2, ensure_ascii=False, default=str)

                logger.info("✅ 大量收集完成!")
                logger.info(f"收集時間: {collection_time:.1f} 秒")
                logger.info(f"有效人物: {len(valid_persons)} 個")
                logger.info(f"資料檔案: {output_file}")

                return {
                    'success': True,
                    'stats': stats_report,
                    'output_file': str(output_file),
                    'valid_persons': len(valid_persons)
                }

        except Exception as e:
            logger.error(f"大量收集失敗: {e}")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """主執行函數"""
    parser = argparse.ArgumentParser(description='大量收集已故人物資料')
    parser.add_argument('--start-year', type=int, default=2010,
                       help='死亡年份起始年 (預設: 2010)')
    parser.add_argument('--end-year', type=int, default=2023,
                       help='死亡年份結束年 (預設: 2023)')
    parser.add_argument('--max-persons', type=int, default=30,
                       help='最多收集人數 (預設: 30)')
    parser.add_argument('--output-prefix', default='mass_collected',
                       help='輸出檔案前綴 (預設: mass_collected)')

    args = parser.parse_args()

    print("壽命預測系統 - 大量已故人物收集")
    print("=" * 50)
    print(f"搜索範圍: {args.start_year}-{args.end_year}")
    print(f"目標收集: {args.max_persons} 人")

    async def run():
        collector = MassDeceasedCollector()
        result = await collector.run_mass_collection(
            start_year=args.start_year,
            end_year=args.end_year,
            max_persons=args.max_persons,
            output_prefix=args.output_prefix
        )

        print("=" * 50)
        if result['success']:
            print("🎉 大量收集成功!")
            print(f"有效人物: {result['valid_persons']}")
            print(f"資料檔案: {result['output_file']}")
            return 0
        else:
            print("❌ 大量收集失敗!")
            print(f"錯誤: {result.get('error', '未知錯誤')}")
            return 1

    try:
        return asyncio.run(run())
    except KeyboardInterrupt:
        print("\n用戶中斷執行")
        return 1
    except Exception as e:
        print(f"\n程序執行失敗: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())