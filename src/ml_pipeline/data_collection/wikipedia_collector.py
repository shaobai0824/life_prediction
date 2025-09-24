"""
維基百科歷史人物資料收集器

收集已故名人的照片、生卒年月等資料用於模型訓練
"""

import requests
import json
import time
import asyncio
import aiohttp
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import re
from dataclasses import dataclass
from urllib.parse import urlparse
import hashlib

import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


@dataclass
class PersonData:
    """歷史人物資料結構"""
    name: str
    birth_date: Optional[date]
    death_date: Optional[date]
    wikipedia_url: str
    photos: List[Dict[str, Any]]
    nationality: Optional[str] = None
    occupation: Optional[str] = None
    cause_of_death: Optional[str] = None

    @property
    def lifespan(self) -> Optional[int]:
        """計算壽命"""
        if self.birth_date and self.death_date:
            return (self.death_date - self.birth_date).days // 365
        return None

    def remaining_lifespan_at_photo(self, photo_date: date) -> Optional[int]:
        """計算照片拍攝時的剩餘壽命"""
        if self.death_date and photo_date:
            if photo_date <= self.death_date:
                return (self.death_date - photo_date).days // 365
        return None


class WikipediaCollector:
    """維基百科資料收集器"""

    def __init__(self, output_dir: str = "./data/collected/wikipedia"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # API 設定
        self.wiki_api_url = "https://en.wikipedia.org/w/api.php"
        self.commons_api_url = "https://commons.wikimedia.org/w/api.php"

        # 請求間隔設定（尊重伺服器）
        self.request_delay = 1.0  # 秒
        self.batch_size = 20

        # 收集統計
        self.stats = {
            'total_searched': 0,
            'valid_persons': 0,
            'photos_collected': 0,
            'training_samples': 0,
            'errors': 0
        }

        # 過濾條件
        self.min_age_at_death = 18  # 最小死亡年齡
        self.max_age_at_death = 100  # 最大死亡年齡
        self.required_photo_years = 5  # 需要至少跨越5年的照片

    async def collect_historical_data(
        self,
        categories: List[str] = None,
        max_persons: int = 1000,
        start_year: int = 1800,
        end_year: int = 2020
    ) -> Dict[str, Any]:
        """收集歷史資料"""
        try:
            logger.info(f"開始收集維基百科歷史人物資料...")

            if categories is None:
                categories = [
                    "Category:20th-century_people",
                    "Category:19th-century_people",
                    "Category:American_politicians",
                    "Category:British_politicians",
                    "Category:American_actors",
                    "Category:British_actors",
                    "Category:Scientists",
                    "Category:Writers",
                    "Category:Artists"
                ]

            # 重置統計
            self.stats = {key: 0 for key in self.stats}
            collected_persons = []

            async with aiohttp.ClientSession() as session:
                for category in categories:
                    logger.info(f"正在處理類別: {category}")

                    # 獲取類別中的人物
                    person_titles = await self._get_category_members(
                        session, category, max_persons // len(categories)
                    )

                    # 批次處理人物資料
                    for i in range(0, len(person_titles), self.batch_size):
                        batch = person_titles[i:i + self.batch_size]
                        batch_results = await self._process_person_batch(
                            session, batch, start_year, end_year
                        )

                        collected_persons.extend(batch_results)

                        # 延遲防止過度請求
                        await asyncio.sleep(self.request_delay)

                        logger.info(f"已處理 {len(collected_persons)} 個人物")

                        if len(collected_persons) >= max_persons:
                            break

                    if len(collected_persons) >= max_persons:
                        break

            # 生成訓練資料集
            training_dataset = await self._generate_training_dataset(collected_persons)

            # 保存結果
            await self._save_collection_results(collected_persons, training_dataset)

            logger.info(f"資料收集完成！收集統計: {self.stats}")

            return {
                'success': True,
                'collected_persons': len(collected_persons),
                'training_samples': len(training_dataset),
                'stats': self.stats,
                'output_dir': str(self.output_dir)
            }

        except Exception as e:
            logger.error(f"資料收集失敗: {e}")
            return {
                'success': False,
                'error': str(e),
                'stats': self.stats
            }

    async def _get_category_members(
        self,
        session: aiohttp.ClientSession,
        category: str,
        limit: int = 100
    ) -> List[str]:
        """獲取類別成員"""
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'categorymembers',
                'cmtitle': category,
                'cmlimit': min(limit, 500),  # API 限制
                'cmtype': 'page',
                'cmnamespace': 0  # 主命名空間
            }

            async with session.get(self.wiki_api_url, params=params) as response:
                data = await response.json()

                if 'query' in data and 'categorymembers' in data['query']:
                    return [member['title'] for member in data['query']['categorymembers']]

                return []

        except Exception as e:
            logger.error(f"獲取類別成員失敗 {category}: {e}")
            return []

    async def _process_person_batch(
        self,
        session: aiohttp.ClientSession,
        titles: List[str],
        start_year: int,
        end_year: int
    ) -> List[PersonData]:
        """批次處理人物資料"""
        results = []

        for title in titles:
            try:
                self.stats['total_searched'] += 1

                # 獲取人物基本資料
                person_info = await self._get_person_info(session, title)
                if not person_info:
                    continue

                # 檢查是否符合條件
                if not self._is_valid_person(person_info, start_year, end_year):
                    continue

                # 獲取照片資料
                photos = await self._get_person_photos(session, title)
                if len(photos) < 2:  # 至少需要2張照片
                    continue

                # 建立人物資料
                person_data = PersonData(
                    name=person_info['name'],
                    birth_date=person_info.get('birth_date'),
                    death_date=person_info.get('death_date'),
                    wikipedia_url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    photos=photos,
                    nationality=person_info.get('nationality'),
                    occupation=person_info.get('occupation'),
                    cause_of_death=person_info.get('cause_of_death')
                )

                results.append(person_data)
                self.stats['valid_persons'] += 1
                self.stats['photos_collected'] += len(photos)

            except Exception as e:
                logger.warning(f"處理人物 {title} 失敗: {e}")
                self.stats['errors'] += 1
                continue

        return results

    async def _get_person_info(
        self,
        session: aiohttp.ClientSession,
        title: str
    ) -> Optional[Dict[str, Any]]:
        """獲取人物基本資訊"""
        try:
            # 獲取頁面內容
            params = {
                'action': 'query',
                'format': 'json',
                'prop': 'extracts|pageprops|categories',
                'exintro': True,
                'explaintext': True,
                'titles': title
            }

            async with session.get(self.wiki_api_url, params=params) as response:
                data = await response.json()

                if 'query' not in data or 'pages' not in data['query']:
                    return None

                page_data = next(iter(data['query']['pages'].values()))
                if 'missing' in page_data:
                    return None

                extract = page_data.get('extract', '')

                # 解析生卒年月
                birth_date, death_date = self._parse_dates_from_text(extract)

                # 檢查是否為已故人員
                if not death_date:
                    return None

                # 提取其他資訊
                nationality = self._extract_nationality(extract)
                occupation = self._extract_occupation(extract)

                return {
                    'name': title,
                    'birth_date': birth_date,
                    'death_date': death_date,
                    'nationality': nationality,
                    'occupation': occupation,
                    'extract': extract
                }

        except Exception as e:
            logger.error(f"獲取人物資訊失敗 {title}: {e}")
            return None

    def _parse_dates_from_text(self, text: str) -> Tuple[Optional[date], Optional[date]]:
        """從文字中解析生卒年月"""
        try:
            # 常見的生卒日期格式
            patterns = [
                r'(\w+ \d{1,2}, \d{4}).*?(\w+ \d{1,2}, \d{4})',  # Month DD, YYYY
                r'(\d{1,2} \w+ \d{4}).*?(\d{1,2} \w+ \d{4})',    # DD Month YYYY
                r'\((\d{4}).*?(\d{4})\)',                        # (YYYY - YYYY)
                r'born.*?(\d{4}).*?died.*?(\d{4})',              # born YYYY died YYYY
                r'(\d{4}).*?(\d{4})',                            # YYYY - YYYY (最寬泛)
            ]

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        birth_str, death_str = match.groups()
                        birth_date = self._parse_date_string(birth_str)
                        death_date = self._parse_date_string(death_str)

                        if birth_date and death_date and birth_date < death_date:
                            return birth_date, death_date
                    except:
                        continue

            return None, None

        except Exception as e:
            logger.error(f"日期解析失敗: {e}")
            return None, None

    def _parse_date_string(self, date_str: str) -> Optional[date]:
        """解析日期字串"""
        try:
            # 提取年份
            year_match = re.search(r'(\d{4})', date_str)
            if not year_match:
                return None

            year = int(year_match.group(1))

            # 嘗試解析完整日期
            month_names = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                'september': 9, 'october': 10, 'november': 11, 'december': 12,
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }

            # 嘗試 Month DD 格式
            month_day_match = re.search(r'(\w+)\s+(\d{1,2})', date_str.lower())
            if month_day_match:
                month_name = month_day_match.group(1)
                day = int(month_day_match.group(2))

                if month_name in month_names:
                    month = month_names[month_name]
                    return date(year, month, day)

            # 嘗試 DD Month 格式
            day_month_match = re.search(r'(\d{1,2})\s+(\w+)', date_str.lower())
            if day_month_match:
                day = int(day_month_match.group(1))
                month_name = day_month_match.group(2)

                if month_name in month_names:
                    month = month_names[month_name]
                    return date(year, month, day)

            # 只有年份，使用年中
            return date(year, 6, 15)

        except Exception as e:
            logger.debug(f"日期解析失敗 {date_str}: {e}")
            return None

    def _extract_nationality(self, text: str) -> Optional[str]:
        """提取國籍"""
        try:
            # 簡單的國籍提取
            nationalities = [
                'American', 'British', 'English', 'German', 'French',
                'Italian', 'Spanish', 'Russian', 'Chinese', 'Japanese',
                'Canadian', 'Australian', 'Dutch', 'Swedish', 'Norwegian'
            ]

            for nationality in nationalities:
                if nationality.lower() in text.lower():
                    return nationality

            return None

        except Exception:
            return None

    def _extract_occupation(self, text: str) -> Optional[str]:
        """提取職業"""
        try:
            # 常見職業關鍵字
            occupations = [
                'politician', 'actor', 'actress', 'scientist', 'writer',
                'author', 'artist', 'musician', 'composer', 'painter',
                'president', 'minister', 'senator', 'congressman',
                'director', 'producer', 'philosopher', 'mathematician'
            ]

            text_lower = text.lower()
            for occupation in occupations:
                if occupation in text_lower:
                    return occupation.title()

            return None

        except Exception:
            return None

    def _is_valid_person(
        self,
        person_info: Dict[str, Any],
        start_year: int,
        end_year: int
    ) -> bool:
        """檢查人物是否符合收集條件"""
        try:
            birth_date = person_info.get('birth_date')
            death_date = person_info.get('death_date')

            if not birth_date or not death_date:
                return False

            # 檢查年份範圍
            if death_date.year < start_year or death_date.year > end_year:
                return False

            # 檢查壽命範圍
            age = (death_date - birth_date).days // 365
            if age < self.min_age_at_death or age > self.max_age_at_death:
                return False

            return True

        except Exception as e:
            logger.debug(f"人物驗證失敗: {e}")
            return False

    async def _get_person_photos(
        self,
        session: aiohttp.ClientSession,
        title: str
    ) -> List[Dict[str, Any]]:
        """獲取人物照片"""
        try:
            # 獲取頁面圖片
            params = {
                'action': 'query',
                'format': 'json',
                'prop': 'images',
                'titles': title,
                'imlimit': 50
            }

            async with session.get(self.wiki_api_url, params=params) as response:
                data = await response.json()

                if 'query' not in data or 'pages' not in data['query']:
                    return []

                page_data = next(iter(data['query']['pages'].values()))
                if 'images' not in page_data:
                    return []

                photos = []
                for image in page_data['images']:
                    image_title = image['title']

                    # 過濾出可能的人像照片
                    if self._is_potential_portrait(image_title):
                        photo_info = await self._get_image_details(session, image_title)
                        if photo_info:
                            photos.append(photo_info)

                return photos[:10]  # 限制每人最多10張照片

        except Exception as e:
            logger.error(f"獲取照片失敗 {title}: {e}")
            return []

    def _is_potential_portrait(self, image_title: str) -> bool:
        """判斷是否可能是人像照片"""
        try:
            title_lower = image_title.lower()

            # 排除明顯非人像的圖片
            exclude_keywords = [
                'signature', 'autograph', 'map', 'chart', 'diagram',
                'logo', 'coat', 'arms', 'flag', 'building', 'grave',
                'monument', 'statue', 'painting', 'drawing'
            ]

            for keyword in exclude_keywords:
                if keyword in title_lower:
                    return False

            # 包含可能的人像關鍵字
            include_keywords = [
                'portrait', 'photo', 'picture', '.jpg', '.jpeg', '.png',
                'headshot', 'face', 'official'
            ]

            for keyword in include_keywords:
                if keyword in title_lower:
                    return True

            # 默認包含（保守策略）
            return True

        except Exception:
            return False

    async def _get_image_details(
        self,
        session: aiohttp.ClientSession,
        image_title: str
    ) -> Optional[Dict[str, Any]]:
        """獲取圖片詳細資訊"""
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'prop': 'imageinfo',
                'titles': image_title,
                'iiprop': 'url|size|timestamp|metadata',
                'iiurlwidth': 512
            }

            async with session.get(self.wiki_api_url, params=params) as response:
                data = await response.json()

                if 'query' not in data or 'pages' not in data['query']:
                    return None

                page_data = next(iter(data['query']['pages'].values()))
                if 'imageinfo' not in page_data or not page_data['imageinfo']:
                    return None

                image_info = page_data['imageinfo'][0]

                # 檢查圖片尺寸
                if image_info.get('width', 0) < 100 or image_info.get('height', 0) < 100:
                    return None

                # 解析拍攝日期
                photo_date = self._parse_image_timestamp(image_info.get('timestamp'))

                return {
                    'title': image_title,
                    'url': image_info.get('url'),
                    'thumb_url': image_info.get('thumburl'),
                    'width': image_info.get('width'),
                    'height': image_info.get('height'),
                    'timestamp': image_info.get('timestamp'),
                    'photo_date': photo_date,
                    'size': image_info.get('size')
                }

        except Exception as e:
            logger.debug(f"獲取圖片詳情失敗 {image_title}: {e}")
            return None

    def _parse_image_timestamp(self, timestamp: str) -> Optional[date]:
        """解析圖片時間戳"""
        try:
            if not timestamp:
                return None

            # 解析 ISO 格式時間戳
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.date()

        except Exception:
            return None

    async def _generate_training_dataset(
        self,
        persons: List[PersonData]
    ) -> List[Dict[str, Any]]:
        """生成訓練資料集"""
        try:
            training_samples = []

            for person in persons:
                if not person.photos or not person.death_date:
                    continue

                for photo in person.photos:
                    if not photo.get('photo_date'):
                        continue

                    # 計算剩餘壽命
                    remaining_years = person.remaining_lifespan_at_photo(photo['photo_date'])
                    if remaining_years is None or remaining_years < 0:
                        continue

                    # 建立訓練樣本
                    sample = {
                        'person_name': person.name,
                        'photo_url': photo['url'],
                        'photo_date': photo['photo_date'].isoformat(),
                        'death_date': person.death_date.isoformat(),
                        'birth_date': person.birth_date.isoformat() if person.birth_date else None,
                        'remaining_lifespan_years': remaining_years,
                        'age_at_photo': None,  # 計算拍照時年齡
                        'total_lifespan': person.lifespan,
                        'nationality': person.nationality,
                        'occupation': person.occupation,
                        'image_width': photo.get('width'),
                        'image_height': photo.get('height'),
                        'wikipedia_url': person.wikipedia_url
                    }

                    # 計算拍照時年齡
                    if person.birth_date:
                        age_at_photo = (photo['photo_date'] - person.birth_date).days // 365
                        sample['age_at_photo'] = age_at_photo

                    training_samples.append(sample)
                    self.stats['training_samples'] += 1

            logger.info(f"生成了 {len(training_samples)} 個訓練樣本")
            return training_samples

        except Exception as e:
            logger.error(f"生成訓練資料集失敗: {e}")
            return []

    async def _save_collection_results(
        self,
        persons: List[PersonData],
        training_dataset: List[Dict[str, Any]]
    ):
        """保存收集結果"""
        try:
            # 保存原始人物資料
            persons_file = self.output_dir / f"persons_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            persons_data = []

            for person in persons:
                person_dict = {
                    'name': person.name,
                    'birth_date': person.birth_date.isoformat() if person.birth_date else None,
                    'death_date': person.death_date.isoformat() if person.death_date else None,
                    'wikipedia_url': person.wikipedia_url,
                    'nationality': person.nationality,
                    'occupation': person.occupation,
                    'lifespan': person.lifespan,
                    'photos': person.photos
                }
                persons_data.append(person_dict)

            with open(persons_file, 'w', encoding='utf-8') as f:
                json.dump(persons_data, f, indent=2, ensure_ascii=False, default=str)

            # 保存訓練資料集
            dataset_file = self.output_dir / f"training_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump(training_dataset, f, indent=2, ensure_ascii=False, default=str)

            # 保存統計報告
            stats_file = self.output_dir / f"collection_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2)

            logger.info(f"結果已保存到: {self.output_dir}")

        except Exception as e:
            logger.error(f"保存結果失敗: {e}")


async def main():
    """測試收集器"""
    collector = WikipediaCollector()

    result = await collector.collect_historical_data(
        categories=[
            "Category:20th-century_American_politicians",
            "Category:American_film_actors"
        ],
        max_persons=50,
        start_year=1950,
        end_year=2020
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())