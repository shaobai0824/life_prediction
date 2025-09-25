#!/usr/bin/env python3
"""
將大量收集的資料轉換為訓練格式

將mass_collect_deceased_persons.py收集的資料轉換為模型訓練所需的格式
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Any
import argparse

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class MassDataConverter:
    """大量資料轉換器"""

    def __init__(self):
        self.project_root = project_root
        self.input_dir = self.project_root / "data" / "collected" / "mass_search"
        self.output_dir = self.project_root / "data" / "collected" / "wikipedia"

    def find_latest_mass_data(self) -> str:
        """尋找最新的大量收集資料檔案"""
        pattern = "mass_collected_*.json"
        files = list(self.input_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(f"找不到大量收集資料檔案，模式: {pattern}")

        # 選擇最新的檔案
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        logger.info(f"找到最新的大量收集檔案: {latest_file}")
        return str(latest_file)

    def convert_to_training_format(self, input_file: str) -> Dict[str, Any]:
        """轉換為訓練格式"""
        try:
            # 讀取大量收集的資料
            with open(input_file, 'r', encoding='utf-8') as f:
                mass_data = json.load(f)

            logger.info(f"載入 {len(mass_data)} 個大量收集的人物資料")

            training_samples = []

            for person in mass_data:
                try:
                    birth_date = datetime.fromisoformat(person['birth_date']).date()
                    death_date = datetime.fromisoformat(person['death_date']).date()

                    # 為每個人的每張照片創建訓練樣本
                    for photo in person.get('photos', []):
                        # 解析照片日期
                        photo_date_str = photo.get('photo_date')
                        if not photo_date_str:
                            continue

                        try:
                            photo_date = datetime.fromisoformat(photo_date_str).date()
                        except:
                            continue

                        # 確保照片日期在合理範圍內
                        if photo_date < birth_date or photo_date > death_date:
                            continue

                        # 計算拍照時的年齡和剩餘壽命
                        age_at_photo = (photo_date - birth_date).days // 365
                        remaining_days = (death_date - photo_date).days
                        remaining_years = remaining_days / 365.25

                        # 過濾不合理的數據
                        if age_at_photo < 0 or remaining_years < 0 or remaining_years > 100:
                            continue

                        # 創建訓練樣本
                        training_sample = {
                            'person_name': person['person_name'],
                            'photo_url': photo['url'],
                            'photo_date': photo_date.isoformat(),
                            'death_date': death_date.isoformat(),
                            'birth_date': birth_date.isoformat(),
                            'remaining_lifespan_years': remaining_years,
                            'age_at_photo': age_at_photo,
                            'total_lifespan': person['age_at_death'],
                            'nationality': person.get('nationality'),
                            'occupation': person.get('occupation'),
                            'image_width': photo.get('width'),
                            'image_height': photo.get('height'),
                            'wikipedia_url': f"https://en.wikipedia.org/wiki/{person['person_name'].replace(' ', '_')}",
                            'source': 'mass_collection'
                        }

                        training_samples.append(training_sample)

                except Exception as e:
                    logger.error(f"處理人物 {person.get('person_name', 'Unknown')} 失敗: {e}")
                    continue

            logger.info(f"成功轉換 {len(training_samples)} 個訓練樣本")

            return {
                'success': True,
                'training_samples': training_samples,
                'original_persons': len(mass_data),
                'converted_samples': len(training_samples)
            }

        except Exception as e:
            logger.error(f"轉換失敗: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def save_training_data(self, training_samples: List[Dict[str, Any]], source_file: str) -> str:
        """保存訓練資料"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存訓練資料
        output_file = self.output_dir / f"mass_training_dataset_{timestamp}.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_samples, f, indent=2, ensure_ascii=False, default=str)

        # 生成轉換報告
        report = {
            'conversion_timestamp': timestamp,
            'source_file': source_file,
            'output_file': str(output_file),
            'statistics': {
                'total_samples': len(training_samples),
                'unique_persons': len(set(sample['person_name'] for sample in training_samples)),
                'age_range': {
                    'min_age': min(sample['age_at_photo'] for sample in training_samples),
                    'max_age': max(sample['age_at_photo'] for sample in training_samples),
                    'avg_age': sum(sample['age_at_photo'] for sample in training_samples) / len(training_samples)
                },
                'remaining_lifespan_range': {
                    'min_remaining': min(sample['remaining_lifespan_years'] for sample in training_samples),
                    'max_remaining': max(sample['remaining_lifespan_years'] for sample in training_samples),
                    'avg_remaining': sum(sample['remaining_lifespan_years'] for sample in training_samples) / len(training_samples)
                }
            }
        }

        report_file = self.output_dir / f"mass_conversion_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"訓練資料已保存至: {output_file}")
        logger.info(f"轉換報告已保存至: {report_file}")

        return str(output_file)

    def run_conversion(self, input_file: str = None) -> Dict[str, Any]:
        """執行轉換"""
        try:
            if input_file is None:
                input_file = self.find_latest_mass_data()

            logger.info(f"開始轉換大量收集資料: {input_file}")

            # 轉換資料
            result = self.convert_to_training_format(input_file)

            if not result['success']:
                return result

            # 保存訓練資料
            output_file = self.save_training_data(result['training_samples'], input_file)

            return {
                'success': True,
                'input_file': input_file,
                'output_file': output_file,
                'statistics': {
                    'original_persons': result['original_persons'],
                    'converted_samples': result['converted_samples'],
                    'conversion_rate': result['converted_samples'] / result['original_persons'] if result['original_persons'] > 0 else 0
                }
            }

        except Exception as e:
            logger.error(f"轉換過程失敗: {e}")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """主執行函數"""
    parser = argparse.ArgumentParser(description='轉換大量收集資料為訓練格式')
    parser.add_argument('--input-file', type=str, help='指定輸入檔案路徑（可選，預設使用最新檔案）')

    args = parser.parse_args()

    print("壽命預測系統 - 大量資料轉換器")
    print("=" * 40)

    try:
        converter = MassDataConverter()
        result = converter.run_conversion(args.input_file)

        print("=" * 40)
        if result['success']:
            print("[SUCCESS] 資料轉換成功!")
            stats = result['statistics']
            print(f"原始人物: {stats['original_persons']} 人")
            print(f"訓練樣本: {stats['converted_samples']} 個")
            print(f"轉換率: {stats['conversion_rate']:.1%}")
            print(f"輸出檔案: {result['output_file']}")
        else:
            print("[ERROR] 資料轉換失敗!")
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