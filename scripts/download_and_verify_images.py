#!/usr/bin/env python3
"""
下載並驗證訓練資料中的圖片

檢查訓練資料中的照片URL是否可以下載，並驗證圖片品質
"""

import sys
import os
import asyncio
import aiohttp
import json
import logging
from pathlib import Path
from datetime import datetime

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from ml_pipeline.data_collection.image_downloader import RealImageDownloader

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ImageVerifier:
    """圖片驗證和下載器"""

    def __init__(self):
        self.image_downloader = RealImageDownloader()
        self.project_root = project_root

    def find_latest_training_data(self) -> str:
        """尋找最新的訓練資料檔案"""
        pattern = "mass_training_dataset_*.json"
        files = list((self.project_root / "data" / "collected" / "wikipedia").glob(pattern))

        if not files:
            raise FileNotFoundError(f"找不到訓練資料檔案")

        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        logger.info(f"使用訓練資料: {latest_file}")
        return str(latest_file)

    async def download_and_verify_images(self, training_file: str):
        """下載並驗證所有圖片"""
        try:
            # 讀取訓練資料
            with open(training_file, 'r', encoding='utf-8') as f:
                training_data = json.load(f)

            logger.info(f"載入 {len(training_data)} 個訓練樣本")

            # 準備下載請求
            download_requests = []
            for sample in training_data:
                download_requests.append({
                    'url': sample['photo_url'],
                    'person_name': sample['person_name'],
                    'photo_date': sample.get('photo_date'),
                    'remaining_years': sample['remaining_lifespan_years']
                })

            logger.info("開始下載和驗證圖片...")

            headers = {
                'User-Agent': 'LifePredictionBot/1.0 (https://example.com/contact; research@example.com) Python/aiohttp',
                'Referer': 'https://en.wikipedia.org/'
            }

            async with aiohttp.ClientSession(headers=headers) as session:
                results = []

                for i, request in enumerate(download_requests, 1):
                    logger.info(f"[{i}/{len(download_requests)}] 處理: {request['person_name']}")

                    # 嘗試下載
                    result = await self.image_downloader.download_image(
                        session=session,
                        image_url=request['url'],
                        person_name=request['person_name'],
                        photo_date=request.get('photo_date')
                    )

                    if result:
                        # 檢查面部可檢測性
                        image_array = self.image_downloader.load_cached_image_as_array(result['cache_path'])
                        if image_array is not None:
                            face_check = self.image_downloader.check_face_detectability(image_array)

                            result.update({
                                'remaining_years': request['remaining_years'],
                                'face_detection': face_check,
                                'download_success': True
                            })

                            if face_check['detectable']:
                                logger.info(f"✅ {request['person_name']} - 下載成功，檢測到 {face_check['faces_detected']} 個面部")
                                logger.info(f"   剩餘壽命: {request['remaining_years']:.1f} 年")
                            else:
                                logger.warning(f"⚠️ {request['person_name']} - 下載成功但無法檢測面部")
                        else:
                            logger.error(f"❌ {request['person_name']} - 圖片載入失敗")
                            result['download_success'] = False
                    else:
                        logger.error(f"❌ {request['person_name']} - 下載失敗")
                        result = {
                            'person_name': request['person_name'],
                            'original_url': request['url'],
                            'remaining_years': request['remaining_years'],
                            'download_success': False
                        }

                    results.append(result)
                    await asyncio.sleep(1)  # 避免過快請求

                return results

        except Exception as e:
            logger.error(f"圖片驗證失敗: {e}")
            return []

    async def generate_report(self, results: list) -> dict:
        """生成下載報告"""
        successful_downloads = [r for r in results if r.get('download_success', False)]
        face_detected = [r for r in successful_downloads if r.get('face_detection', {}).get('detectable', False)]

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(results),
            'successful_downloads': len(successful_downloads),
            'face_detected': len(face_detected),
            'download_success_rate': len(successful_downloads) / len(results) if results else 0,
            'face_detection_rate': len(face_detected) / len(successful_downloads) if successful_downloads else 0,
            'cache_stats': self.image_downloader.get_cache_statistics(),
            'detailed_results': []
        }

        # 詳細結果
        for result in results:
            detail = {
                'person_name': result['person_name'],
                'remaining_years': result.get('remaining_years', 0),
                'download_success': result.get('download_success', False),
                'face_detected': result.get('face_detection', {}).get('detectable', False),
                'faces_count': result.get('face_detection', {}).get('faces_detected', 0)
            }
            report['detailed_results'].append(detail)

        return report

    async def run_verification(self):
        """執行完整驗證流程"""
        try:
            # 尋找訓練資料
            training_file = self.find_latest_training_data()

            # 下載和驗證圖片
            results = await self.download_and_verify_images(training_file)

            # 生成報告
            report = await self.generate_report(results)

            # 保存報告
            report_file = self.project_root / "output" / f"image_verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_file.parent.mkdir(exist_ok=True)

            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            # 輸出總結
            print(f"\n圖片驗證完成!")
            print(f"總樣本數: {report['total_samples']}")
            print(f"成功下載: {report['successful_downloads']} ({report['download_success_rate']:.1%})")
            print(f"檢測到面部: {report['face_detected']} ({report['face_detection_rate']:.1%})")
            print(f"快取檔案數: {report['cache_stats']['total_files']}")
            print(f"快取大小: {report['cache_stats']['total_size_mb']} MB")
            print(f"詳細報告: {report_file}")

            return report

        except Exception as e:
            logger.error(f"驗證流程失敗: {e}")
            return None


async def main():
    """主執行函數"""
    print("壽命預測系統 - 圖片下載驗證")
    print("=" * 40)

    verifier = ImageVerifier()
    await verifier.run_verification()


if __name__ == "__main__":
    asyncio.run(main())