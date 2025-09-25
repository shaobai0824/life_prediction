#!/usr/bin/env python3
"""
面部檢測測試腳本

測試OpenCV Haar Cascade的面部檢測功能
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import logging

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from ml_pipeline.data_collection.image_downloader import RealImageDownloader

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_face_image() -> np.ndarray:
    """創建測試用的面部圖片 - 優化Haar Cascade檢測"""
    # 創建一個符合Haar Cascade特徵的面部圖片
    image = np.ones((300, 300, 3), dtype=np.uint8) * 200

    # 面部區域 - 更符合Haar特徵的矩形區域
    face_x, face_y = 50, 50
    face_w, face_h = 200, 200

    # 畫臉部主區域
    cv2.rectangle(image, (face_x, face_y), (face_x + face_w, face_y + face_h), (220, 200, 180), -1)

    # 眼睛區域 - Haar Cascade檢測眼睛的對比度
    eye_y = face_y + 60
    eye1_x = face_x + 40
    eye2_x = face_x + 140

    # 左眼
    cv2.rectangle(image, (eye1_x, eye_y), (eye1_x + 40, eye_y + 20), (180, 160, 140), -1)
    cv2.rectangle(image, (eye1_x + 5, eye_y + 5), (eye1_x + 35, eye_y + 15), (50, 50, 50), -1)

    # 右眼
    cv2.rectangle(image, (eye2_x, eye_y), (eye2_x + 40, eye_y + 20), (180, 160, 140), -1)
    cv2.rectangle(image, (eye2_x + 5, eye_y + 5), (eye2_x + 35, eye_y + 15), (50, 50, 50), -1)

    # 鼻子區域
    nose_x = face_x + 90
    nose_y = face_y + 100
    cv2.rectangle(image, (nose_x, nose_y), (nose_x + 20, nose_y + 40), (200, 180, 160), -1)

    # 嘴巴區域
    mouth_x = face_x + 70
    mouth_y = face_y + 160
    cv2.rectangle(image, (mouth_x, mouth_y), (mouth_x + 60, mouth_y + 20), (160, 100, 100), -1)

    # 增加更多對比度特徵幫助檢測
    # 眉毛區域
    cv2.rectangle(image, (eye1_x, eye_y - 15), (eye1_x + 40, eye_y - 10), (100, 80, 60), -1)
    cv2.rectangle(image, (eye2_x, eye_y - 15), (eye2_x + 40, eye_y - 10), (100, 80, 60), -1)

    return image


def test_haar_cascade_detection():
    """測試Haar Cascade面部檢測"""
    logger.info("測試Haar Cascade面部檢測...")

    try:
        # 載入Haar Cascade檢測器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if face_cascade.empty():
            logger.error("❌ 無法載入Haar Cascade檢測器")
            return False

        # 創建測試圖片
        test_image = create_test_face_image()

        # 轉換為灰階
        gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

        # 檢測面部
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        logger.info(f"檢測到 {len(faces)} 個面部")

        if len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                logger.info(f"  面部 {i+1}: ({x}, {y}, {w}, {h}) 面積: {w*h}")

            # 保存檢測結果圖片（可選）
            result_image = test_image.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            output_dir = project_root / "output"
            output_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(output_dir / "face_detection_test.jpg"), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            logger.info(f"檢測結果圖片已保存至: {output_dir / 'face_detection_test.jpg'}")

            return True
        else:
            logger.warning("⚠️ 未檢測到面部")
            return False

    except Exception as e:
        logger.error(f"❌ 面部檢測測試失敗: {e}")
        return False


def test_image_downloader():
    """測試圖片下載器的面部檢測功能"""
    logger.info("測試圖片下載器的面部檢測功能...")

    try:
        downloader = RealImageDownloader()

        # 創建測試圖片
        test_image = create_test_face_image()

        # 檢查面部檢測
        face_check = downloader.check_face_detectability(test_image)

        logger.info("面部檢測結果:")
        logger.info(f"  檢測到面部數量: {face_check['faces_detected']}")
        logger.info(f"  是否可檢測: {face_check['detectable']}")
        logger.info(f"  最大面部面積: {face_check['largest_face_area']}")

        if face_check.get('error'):
            logger.error(f"  錯誤: {face_check['error']}")

        return face_check['detectable']

    except Exception as e:
        logger.error(f"❌ 圖片下載器測試失敗: {e}")
        return False


def test_cache_statistics():
    """測試快取統計功能"""
    logger.info("測試快取統計功能...")

    try:
        downloader = RealImageDownloader()
        stats = downloader.get_cache_statistics()

        logger.info("快取統計:")
        logger.info(f"  快取檔案數: {stats['total_files']}")
        logger.info(f"  總大小: {stats['total_size_mb']} MB")
        logger.info(f"  快取目錄: {stats['cache_dir']}")

        if stats.get('error'):
            logger.warning(f"  統計錯誤: {stats['error']}")

        return True

    except Exception as e:
        logger.error(f"❌ 快取統計測試失敗: {e}")
        return False


def main():
    """主執行函數"""
    print("壽命預測系統 - 面部檢測測試")
    print("=" * 40)

    tests = [
        ("Haar Cascade檢測", test_haar_cascade_detection),
        ("圖片下載器面部檢測", test_image_downloader),
        ("快取統計", test_cache_statistics)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n[TEST] 執行測試: {test_name}")
        try:
            if test_func():
                print(f"[PASS] {test_name} - 通過")
                passed += 1
            else:
                print(f"[FAIL] {test_name} - 失敗")
        except Exception as e:
            print(f"[ERROR] {test_name} - 異常: {e}")

    print(f"\n" + "=" * 40)
    print(f"測試結果: {passed}/{total} 通過")

    if passed == total:
        print("[SUCCESS] 所有面部檢測測試通過！")
        return 0
    else:
        print("[WARNING] 部分測試失敗")
        return 1


if __name__ == "__main__":
    sys.exit(main())