"""
真實圖片下載和驗證模組

負責從Wikipedia等來源下載真實圖片並進行品質驗證
"""

import asyncio
import aiohttp
from pathlib import Path
import hashlib
import logging
from typing import Optional, Dict, Any, List
import cv2
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)


class RealImageDownloader:
    """真實圖片下載器"""

    def __init__(self, cache_dir: str = "data/images/cache", max_image_size: int = 5 * 1024 * 1024):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_image_size = max_image_size
        self.min_dimension = 100
        self.max_dimension = 2048

    def _generate_cache_filename(self, url: str, person_name: str, photo_date: str = None) -> str:
        """生成快取檔案名"""
        # 使用URL哈希確保唯一性
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]

        # 清理人名作為檔案名
        clean_name = "".join(c if c.isalnum() or c in ".-_" else "_" for c in person_name)

        # 添加日期（如果有）
        date_part = f"_{photo_date[:4]}" if photo_date else ""

        return f"{clean_name}{date_part}_{url_hash}.jpg"

    async def download_image(
        self,
        session: aiohttp.ClientSession,
        image_url: str,
        person_name: str,
        photo_date: str = None
    ) -> Optional[Dict[str, Any]]:
        """下載並驗證圖片"""
        try:
            # 生成快取檔案路徑
            cache_filename = self._generate_cache_filename(image_url, person_name, photo_date)
            cache_path = self.cache_dir / cache_filename

            # 檢查快取是否存在
            if cache_path.exists():
                logger.debug(f"使用快取圖片: {cache_filename}")
                return await self._validate_cached_image(cache_path, image_url, person_name)

            # 下載圖片
            logger.info(f"下載圖片: {person_name} - {image_url}")

            async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    logger.warning(f"圖片下載失敗 {response.status}: {image_url}")
                    return None

                # 檢查內容大小
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.max_image_size:
                    logger.warning(f"圖片過大 ({content_length} bytes): {image_url}")
                    return None

                # 讀取圖片數據
                image_data = await response.read()

                if len(image_data) > self.max_image_size:
                    logger.warning(f"圖片數據過大 ({len(image_data)} bytes): {image_url}")
                    return None

            # 驗證圖片格式和品質
            validation_result = await self._validate_image_data(image_data, image_url)
            if not validation_result['valid']:
                logger.warning(f"圖片驗證失敗: {validation_result['reason']}")
                return None

            # 保存到快取
            try:
                with open(cache_path, 'wb') as f:
                    f.write(image_data)
                logger.debug(f"圖片已保存至快取: {cache_filename}")
            except Exception as e:
                logger.warning(f"保存快取失敗: {e}")

            # 返回驗證結果
            return {
                'cache_path': str(cache_path),
                'cache_filename': cache_filename,
                'original_url': image_url,
                'person_name': person_name,
                'photo_date': photo_date,
                'file_size': len(image_data),
                'dimensions': validation_result['dimensions'],
                'format': validation_result['format']
            }

        except asyncio.TimeoutError:
            logger.warning(f"圖片下載超時: {image_url}")
            return None
        except Exception as e:
            logger.error(f"圖片下載異常 {image_url}: {e}")
            return None

    async def _validate_image_data(self, image_data: bytes, image_url: str) -> Dict[str, Any]:
        """驗證圖片數據"""
        try:
            # 使用PIL驗證圖片格式
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size
            image_format = image.format

            # 檢查尺寸
            if width < self.min_dimension or height < self.min_dimension:
                return {
                    'valid': False,
                    'reason': f'圖片尺寸過小: {width}x{height}'
                }

            if width > self.max_dimension or height > self.max_dimension:
                return {
                    'valid': False,
                    'reason': f'圖片尺寸過大: {width}x{height}'
                }

            # 檢查格式
            if image_format not in ['JPEG', 'PNG', 'WebP']:
                return {
                    'valid': False,
                    'reason': f'不支援的圖片格式: {image_format}'
                }

            # 檢查圖片是否損壞
            image.verify()

            return {
                'valid': True,
                'dimensions': (width, height),
                'format': image_format,
                'reason': 'valid'
            }

        except Exception as e:
            return {
                'valid': False,
                'reason': f'圖片驗證失敗: {str(e)}'
            }

    async def _validate_cached_image(self, cache_path: Path, original_url: str, person_name: str) -> Optional[Dict[str, Any]]:
        """驗證快取圖片"""
        try:
            if not cache_path.exists() or cache_path.stat().st_size == 0:
                return None

            # 讀取並驗證快取圖片
            with open(cache_path, 'rb') as f:
                image_data = f.read()

            validation_result = await self._validate_image_data(image_data, original_url)
            if not validation_result['valid']:
                # 刪除損壞的快取
                cache_path.unlink(missing_ok=True)
                return None

            return {
                'cache_path': str(cache_path),
                'cache_filename': cache_path.name,
                'original_url': original_url,
                'person_name': person_name,
                'file_size': len(image_data),
                'dimensions': validation_result['dimensions'],
                'format': validation_result['format']
            }

        except Exception as e:
            logger.error(f"快取圖片驗證失敗 {cache_path}: {e}")
            return None

    def load_cached_image_as_array(self, cache_path: str) -> Optional[np.ndarray]:
        """載入快取圖片為numpy陣列"""
        try:
            # 使用OpenCV讀取圖片
            image = cv2.imread(cache_path)
            if image is None:
                return None

            # 轉換色彩空間 BGR -> RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image_rgb

        except Exception as e:
            logger.error(f"載入圖片失敗 {cache_path}: {e}")
            return None

    def check_face_detectability(self, image_array: np.ndarray) -> Dict[str, Any]:
        """檢查圖片中是否可檢測到面部（使用Haar Cascade）"""
        try:
            # 轉換為灰階
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

            # 載入Haar Cascade面部檢測器
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # 檢測面部
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )

            return {
                'faces_detected': len(faces),
                'face_boxes': faces.tolist() if len(faces) > 0 else [],
                'detectable': len(faces) > 0,
                'largest_face_area': max([w * h for (x, y, w, h) in faces]) if len(faces) > 0 else 0
            }

        except Exception as e:
            logger.error(f"面部檢測失敗: {e}")
            return {
                'faces_detected': 0,
                'face_boxes': [],
                'detectable': False,
                'largest_face_area': 0,
                'error': str(e)
            }

    async def batch_download_images(
        self,
        session: aiohttp.ClientSession,
        image_requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """批次下載圖片"""
        results = []

        for request in image_requests:
            result = await self.download_image(
                session=session,
                image_url=request['url'],
                person_name=request['person_name'],
                photo_date=request.get('photo_date')
            )

            if result:
                # 檢查面部可檢測性
                image_array = self.load_cached_image_as_array(result['cache_path'])
                if image_array is not None:
                    face_check = self.check_face_detectability(image_array)
                    result['face_detection'] = face_check

                    if face_check['detectable']:
                        results.append(result)
                        logger.info(f"✅ {request['person_name']} - 圖片下載並驗證成功")
                    else:
                        logger.warning(f"⚠️ {request['person_name']} - 圖片無法檢測到面部")
                        # 刪除無面部的圖片快取
                        Path(result['cache_path']).unlink(missing_ok=True)
                else:
                    logger.warning(f"⚠️ {request['person_name']} - 圖片載入失敗")
            else:
                logger.warning(f"❌ {request['person_name']} - 圖片下載失敗")

            # 避免過快請求
            await asyncio.sleep(1)

        return results

    def get_cache_statistics(self) -> Dict[str, Any]:
        """獲取快取統計"""
        try:
            cache_files = list(self.cache_dir.glob('*.jpg')) + list(self.cache_dir.glob('*.png'))
            total_files = len(cache_files)
            total_size = sum(f.stat().st_size for f in cache_files if f.exists())

            return {
                'total_files': total_files,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'cache_dir': str(self.cache_dir)
            }
        except Exception as e:
            logger.error(f"獲取快取統計失敗: {e}")
            return {
                'total_files': 0,
                'total_size_bytes': 0,
                'total_size_mb': 0,
                'cache_dir': str(self.cache_dir),
                'error': str(e)
            }