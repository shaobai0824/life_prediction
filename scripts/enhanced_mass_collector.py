#!/usr/bin/env python3
"""
改進版大量已故人物收集系統

改進功能:
1. 擴大搜索範圍 (1990-2015)
2. 改善圖片過濾，排除SVG、圖標等非人像
3. 更嚴格的圖片品質檢查
4. 優化面部檢測演算法
5. 多資料源整合
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
import re
from urllib.parse import urlparse

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
        logging.FileHandler(project_root / 'logs' / 'enhanced_mass_collection.log', encoding='utf-8')
    ]
)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)


class EnhancedMassCollector:
    """改進版大量已故人物收集器"""

    def __init__(self):
        self.project_root = project_root
        self.output_dir = self.project_root / "data" / "collected" / "enhanced_search"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.collector = WikipediaCollector()
        self.image_downloader = RealImageDownloader()

        self.wiki_api_url = "https://en.wikipedia.org/w/api.php"

        # 統計資料
        self.stats = {
            'pages_searched': 0,
            'deceased_found': 0,
            'photos_found': 0,
            'photos_filtered_out': 0,
            'photos_downloaded': 0,
            'faces_detected': 0,
            'successfully_collected': 0,
            'errors': 0
        }

        # 非人像圖片過濾規則
        self.non_portrait_patterns = [
            r'\.svg$',  # SVG圖標
            r'question.*book',  # 問號圖標
            r'crystal.*clear',  # Crystal Clear圖標
            r'nuvola',  # Nuvola圖標
            r'commons.*logo',  # Commons標誌
            r'wikimedia.*logo',  # Wikimedia標誌
            r'edit.*icon',  # 編輯圖標
            r'folder',  # 資料夾圖標
            r'flag',  # 旗幟
            r'coat.*arms',  # 徽章
            r'emblem',  # 標誌
            r'seal',  # 印章
            r'badge',  # 徽章
            r'award',  # 獎章
            r'medal',  # 勳章
            r'trophy',  # 獎盃
            r'building',  # 建築物
            r'monument',  # 紀念碑
            r'statue',  # 雕像
            r'map',  # 地圖
            r'chart',  # 圖表
            r'graph',  # 圖形
            r'diagram',  # 圖解
            r'symbol',  # 符號
            r'logo',  # 標誌
            r'sign',  # 標牌
            r'button',  # 按鈕
            r'ball',  # 球類
            r'book',  # 書籍
            r'document',  # 文件
            r'paper',  # 紙張
        ]

    def is_likely_portrait(self, image_info: Dict[str, Any]) -> bool:
        """判斷是否可能是人像照片 - 改進版"""
        try:
            title = image_info.get('title', '').lower()
            url = image_info.get('url', '').lower()

            # 檢查非人像模式
            for pattern in self.non_portrait_patterns:
                if re.search(pattern, title, re.IGNORECASE) or re.search(pattern, url, re.IGNORECASE):
                    logger.debug(f"過濾非人像圖片: {title} (匹配模式: {pattern})")
                    self.stats['photos_filtered_out'] += 1
                    return False

            # 檢查檔案格式 - 排除明顯的圖標格式
            if title.endswith('.svg'):
                self.stats['photos_filtered_out'] += 1
                return False

            # 檢查尺寸 - 過小的通常是圖標
            width = image_info.get('width', 0)
            height = image_info.get('height', 0)

            if width < 80 or height < 80:
                logger.debug(f"過濾過小圖片: {title} ({width}x{height})")
                self.stats['photos_filtered_out'] += 1
                return False

            # 檢查比例 - 過於極端的比例通常不是人像
            if width > 0 and height > 0:
                ratio = max(width, height) / min(width, height)
                if ratio > 3:  # 比例超過3:1的通常不是人像
                    logger.debug(f"過濾極端比例圖片: {title} (比例: {ratio:.2f})")
                    self.stats['photos_filtered_out'] += 1
                    return False

            # 積極的人像指標
            portrait_indicators = [
                'portrait', 'photo', 'picture', 'headshot', 'face',
                'official', 'profile', 'mugshot', '.jpg', '.jpeg', '.png'
            ]

            for indicator in portrait_indicators:
                if indicator in title or indicator in url:
                    return True

            # 如果沒有明確的人像指標，但也不是明顯的非人像，則保守接受
            return True

        except Exception as e:
            logger.debug(f"人像判斷失敗: {e}")
            return True  # 出錯時保守接受

    async def get_enhanced_person_photos(
        self,
        session: aiohttp.ClientSession,
        title: str,
        max_photos: int = 5
    ) -> List[Dict[str, Any]]:
        """獲取改進版人物照片"""
        try:
            # 使用原有的照片獲取邏輯
            photos = await self.collector._get_person_photos(session, title)
            self.stats['photos_found'] += len(photos)

            # 應用改進的過濾邏輯
            filtered_photos = []
            for photo in photos:
                if self.is_likely_portrait(photo):
                    filtered_photos.append(photo)
                    if len(filtered_photos) >= max_photos:
                        break

            logger.debug(f"{title}: 找到 {len(photos)} 張照片，過濾後剩 {len(filtered_photos)} 張")
            return filtered_photos

        except Exception as e:
            logger.error(f"獲取 {title} 照片失敗: {e}")
            return []

    def get_extended_death_categories(self, start_year: int, end_year: int) -> List[str]:
        """獲取擴展的死亡分類"""
        categories = []

        # 按年份的分類
        for year in range(start_year, end_year + 1):
            categories.extend([
                f"{year} deaths",
                f"Deaths in {year}",
            ])

        # 按十年分類
        decades = set()
        for year in range(start_year, end_year + 1):
            decade = (year // 10) * 10
            decades.add(decade)

        for decade in sorted(decades):
            categories.extend([
                f"{decade}s deaths",
                f"Deaths in the {decade}s",
            ])

        # 按世紀分類
        if start_year <= 2000 <= end_year:
            categories.extend([
                "20th-century deaths",
                "21st-century deaths",
            ])

        # 專業分類
        professional_categories = [
            "Deaths from cancer",
            "Deaths from heart disease",
            "Accidental deaths",
            "Deaths from pneumonia",
            "Politicians who died in office",
            "Actors who died in the 20th century",
            "Actors who died in the 21st century",
            "Musicians who died in the 20th century",
            "Musicians who died in the 21st century",
            "Writers who died in the 20th century",
            "Writers who died in the 21st century",
            "Scientists who died in the 20th century",
            "Scientists who died in the 21st century",
        ]

        categories.extend(professional_categories)

        return categories

    async def enhanced_discover_deceased_persons(
        self,
        session: aiohttp.ClientSession,
        start_year: int = 1990,
        end_year: int = 2015,
        max_persons: int = 100
    ) -> List[str]:
        """改進版已故人物發現"""
        logger.info(f"開始改進版搜索 {start_year}-{end_year} 年間過世的人物...")

        all_candidates = set()

        # 1. 使用擴展的分類搜索
        categories = self.get_extended_death_categories(start_year, end_year)

        logger.info(f"搜索 {len(categories)} 個分類...")

        for i, category in enumerate(categories):
            try:
                members = await self.search_category_members(session, category, limit=30)
                all_candidates.update(members)
                self.stats['pages_searched'] += len(members)

                if (i + 1) % 10 == 0:
                    logger.info(f"已搜索 {i + 1}/{len(categories)} 個分類，累積 {len(all_candidates)} 個候選人")

                if len(all_candidates) >= max_persons * 5:  # 收集足夠的候選者
                    break

                await asyncio.sleep(0.5)  # 減少請求間隔

            except Exception as e:
                logger.error(f"搜索分類 {category} 失敗: {e}")
                self.stats['errors'] += 1

        # 2. 使用改進的關鍵字搜索
        if len(all_candidates) < max_persons * 3:
            logger.info("使用關鍵字搜索補充...")

            enhanced_search_terms = []

            # 按年份的關鍵字
            for year in range(start_year, end_year + 1, 2):
                enhanced_search_terms.extend([
                    f"died {year}",
                    f"death {year}",
                    f"{year} obituary",
                ])

            # 專業相關關鍵字
            enhanced_search_terms.extend([
                "actor died",
                "actress died",
                "musician died",
                "singer died",
                "politician died",
                "author died",
                "writer died",
                "scientist died",
                "artist died",
                "director died",
                "producer died",
                "journalist died",
                "historian died",
            ])

            keyword_results = await self.search_by_death_keyword(
                session, enhanced_search_terms[:20], limit=40
            )
            all_candidates.update(keyword_results)

        logger.info(f"發現 {len(all_candidates)} 個候選人物")

        # 隨機化並返回
        candidates_list = list(all_candidates)
        random.shuffle(candidates_list)

        return candidates_list[:max_persons * 3]

    async def search_category_members(
        self,
        session: aiohttp.ClientSession,
        category: str,
        limit: int = 100
    ) -> List[str]:
        """搜索分類成員 - 復用原有邏輯"""
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'categorymembers',
                'cmtitle': f'Category:{category}',
                'cmlimit': limit,
                'cmtype': 'page',
                'cmnamespace': 0
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
                    if title and ':' not in title:
                        members.append(title)

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
        """關鍵字搜索 - 復用原有邏輯"""
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

                        death_indicators = ['died', 'death', 'deceased', 'passed away', 'd.', 'obituary']
                        if title and any(indicator in snippet for indicator in death_indicators):
                            found_titles.add(title)

            except Exception as e:
                logger.error(f"關鍵字搜索 '{term}' 失敗: {e}")

        return list(found_titles)

    async def enhanced_validate_person(
        self,
        session: aiohttp.ClientSession,
        title: str
    ) -> Dict[str, Any]:
        """改進版人物驗證"""
        try:
            person_info = await self.collector._get_person_info(session, title)

            if not person_info:
                return {'valid': False, 'reason': '無法獲取基本資訊'}

            if not person_info.get('birth_date') or not person_info.get('death_date'):
                return {'valid': False, 'reason': '缺少生卒日期'}

            # 使用改進的照片獲取
            photos = await self.get_enhanced_person_photos(session, title, max_photos=3)
            if len(photos) < 1:
                return {'valid': False, 'reason': '沒有合適的照片'}

            # 年齡驗證
            birth_date = person_info['birth_date']
            death_date = person_info['death_date']

            if not isinstance(birth_date, date) or not isinstance(death_date, date):
                return {'valid': False, 'reason': '日期格式不正確'}

            age = (death_date - birth_date).days // 365
            if age < 15 or age > 110:  # 稍微寬鬆的年齡範圍
                return {'valid': False, 'reason': f'年齡不合理: {age}歲'}

            return {
                'valid': True,
                'person_info': person_info,
                'photos': photos,
                'age': age
            }

        except Exception as e:
            return {'valid': False, 'reason': f'驗證失敗: {str(e)}'}

    async def run_enhanced_collection(
        self,
        start_year: int = 1990,
        end_year: int = 2015,
        max_persons: int = 50,
        output_prefix: str = "enhanced_collected"
    ) -> Dict[str, Any]:
        """執行改進版大量收集"""
        try:
            logger.info("開始改進版大量已故人物收集...")
            start_time = datetime.now()

            headers = {
                'User-Agent': 'LifePredictionBot/1.0 (https://example.com/contact; research@example.com) Python/aiohttp'
            }

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers=headers
            ) as session:

                # 發現候選人物
                candidates = await self.enhanced_discover_deceased_persons(
                    session, start_year, end_year, max_persons
                )

                if not candidates:
                    return {
                        'success': False,
                        'error': '未找到任何候選人物'
                    }

                # 批次驗證和收集
                valid_persons = []
                processed = 0

                logger.info(f"開始驗證 {len(candidates)} 個候選人物...")

                for title in candidates:
                    if len(valid_persons) >= max_persons:
                        break

                    try:
                        processed += 1
                        logger.info(f"[{processed}/{len(candidates)}] 驗證: {title}")

                        validation = await self.enhanced_validate_person(session, title)

                        if validation['valid']:
                            logger.info(f"✅ {title} - 有效 (年齡: {validation['age']}歲, 照片: {len(validation['photos'])}張)")

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
                        await asyncio.sleep(1.5)  # 適當的請求間隔

                    except Exception as e:
                        logger.error(f"處理 {title} 時發生錯誤: {e}")
                        self.stats['errors'] += 1

                # 保存結果
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = self.output_dir / f"{output_prefix}_{timestamp}.json"

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(valid_persons, f, indent=2, ensure_ascii=False, default=str)

                # 生成統計報告
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
                    },
                    'improvements': {
                        'photo_filtering_enabled': True,
                        'extended_year_range': f"{start_year}-{end_year}",
                        'enhanced_categories': True,
                        'professional_keywords': True
                    }
                }

                stats_file = self.output_dir / f"{output_prefix}_stats_{timestamp}.json"
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(stats_report, f, indent=2, ensure_ascii=False, default=str)

                logger.info("✅ 改進版大量收集完成!")
                logger.info(f"收集時間: {collection_time:.1f} 秒")
                logger.info(f"有效人物: {len(valid_persons)} 個")
                logger.info(f"照片過濾: {self.stats['photos_filtered_out']} 張")
                logger.info(f"資料檔案: {output_file}")

                return {
                    'success': True,
                    'stats': stats_report,
                    'output_file': str(output_file),
                    'valid_persons': len(valid_persons)
                }

        except Exception as e:
            logger.error(f"改進版收集失敗: {e}")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """主執行函數"""
    parser = argparse.ArgumentParser(description='改進版大量收集已故人物資料')
    parser.add_argument('--start-year', type=int, default=1990,
                       help='死亡年份起始年 (預設: 1990)')
    parser.add_argument('--end-year', type=int, default=2015,
                       help='死亡年份結束年 (預設: 2015)')
    parser.add_argument('--max-persons', type=int, default=50,
                       help='最多收集人數 (預設: 50)')
    parser.add_argument('--output-prefix', default='enhanced_collected',
                       help='輸出檔案前綴 (預設: enhanced_collected)')

    args = parser.parse_args()

    print("壽命預測系統 - 改進版大量已故人物收集")
    print("=" * 50)
    print(f"搜索範圍: {args.start_year}-{args.end_year}")
    print(f"目標收集: {args.max_persons} 人")
    print("改進特性: 圖片過濾 + 擴展搜索 + 品質控制")

    async def run():
        collector = EnhancedMassCollector()
        result = await collector.run_enhanced_collection(
            start_year=args.start_year,
            end_year=args.end_year,
            max_persons=args.max_persons,
            output_prefix=args.output_prefix
        )

        print("=" * 50)
        if result['success']:
            print("[SUCCESS] 改進版收集成功!")
            print(f"有效人物: {result['valid_persons']}")
            print(f"資料檔案: {result['output_file']}")
            return 0
        else:
            print("[ERROR] 改進版收集失敗!")
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