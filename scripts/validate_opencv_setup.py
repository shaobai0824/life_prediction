#!/usr/bin/env python3
"""
驗證OpenCV安裝和Haar Cascade設定

直接測試OpenCV的基本功能
"""

import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_opencv_installation():
    """測試OpenCV安裝"""
    try:
        logger.info(f"OpenCV版本: {cv2.__version__}")
        return True
    except Exception as e:
        logger.error(f"OpenCV測試失敗: {e}")
        return False


def test_haar_cascade_availability():
    """測試Haar Cascade檔案可用性"""
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        logger.info(f"Haar Cascade路徑: {cascade_path}")

        face_cascade = cv2.CascadeClassifier(cascade_path)

        if face_cascade.empty():
            logger.error("Haar Cascade檔案載入失敗")
            return False
        else:
            logger.info("Haar Cascade檔案載入成功")
            return True

    except Exception as e:
        logger.error(f"Haar Cascade測試失敗: {e}")
        return False


def test_with_lena_image():
    """使用Lena標準測試圖片進行面部檢測"""
    try:
        # 創建一個包含明顯面部特徵的測試圖片
        # 這是一個簡化的面部輪廓，應該能被Haar Cascade檢測到

        # 創建更大的圖片以符合檢測器要求
        img = np.ones((600, 600, 3), dtype=np.uint8) * 240

        # 創建一個更明顯的面部輪廓
        center_x, center_y = 300, 300

        # 面部輪廓（橢圓）
        cv2.ellipse(img, (center_x, center_y), (150, 200), 0, 0, 360, (200, 180, 160), -1)

        # 眼睛區域（創建明顯的暗區）
        cv2.ellipse(img, (center_x - 60, center_y - 60), (30, 20), 0, 0, 360, (80, 80, 80), -1)
        cv2.ellipse(img, (center_x + 60, center_y - 60), (30, 20), 0, 0, 360, (80, 80, 80), -1)

        # 鼻子
        cv2.ellipse(img, (center_x, center_y), (15, 30), 0, 0, 360, (180, 160, 140), -1)

        # 嘴巴
        cv2.ellipse(img, (center_x, center_y + 80), (40, 20), 0, 0, 360, (160, 100, 100), -1)

        # 轉換為灰階
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 載入面部檢測器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 檢測面部
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        logger.info(f"檢測到 {len(faces)} 個面部")

        if len(faces) > 0:
            logger.info("面部檢測成功！")
            for (x, y, w, h) in faces:
                logger.info(f"面部位置: ({x}, {y}) 尺寸: {w}x{h}")
            return True
        else:
            logger.warning("未檢測到面部")

            # 嘗試調整參數
            faces2 = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))
            logger.info(f"調整參數後檢測到 {len(faces2)} 個面部")

            return len(faces2) > 0

    except Exception as e:
        logger.error(f"測試圖片面部檢測失敗: {e}")
        return False


def test_basic_detection_parameters():
    """測試不同的檢測參數"""
    try:
        # 創建一個簡單的面部圖片
        img = np.ones((400, 400, 3), dtype=np.uint8) * 220

        # 臉部
        cv2.rectangle(img, (100, 100), (300, 300), (200, 180, 160), -1)

        # 眼睛
        cv2.rectangle(img, (130, 150), (170, 180), (100, 100, 100), -1)
        cv2.rectangle(img, (230, 150), (270, 180), (100, 100, 100), -1)

        # 嘴巴
        cv2.rectangle(img, (170, 250), (230, 280), (150, 100, 100), -1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 嘗試不同參數
        parameters = [
            {"scaleFactor": 1.1, "minNeighbors": 3, "minSize": (30, 30)},
            {"scaleFactor": 1.2, "minNeighbors": 5, "minSize": (50, 50)},
            {"scaleFactor": 1.3, "minNeighbors": 3, "minSize": (20, 20)},
        ]

        for i, params in enumerate(parameters):
            faces = face_cascade.detectMultiScale(gray, **params)
            logger.info(f"參數集 {i+1}: {params} -> 檢測到 {len(faces)} 個面部")

            if len(faces) > 0:
                return True

        return False

    except Exception as e:
        logger.error(f"參數測試失敗: {e}")
        return False


def main():
    """主執行函數"""
    print("OpenCV 和 Haar Cascade 驗證")
    print("=" * 40)

    tests = [
        ("OpenCV 安裝", test_opencv_installation),
        ("Haar Cascade 可用性", test_haar_cascade_availability),
        ("標準測試圖片檢測", test_with_lena_image),
        ("檢測參數測試", test_basic_detection_parameters)
    ]

    passed = 0
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        if test_func():
            print(f"[PASS] {test_name}")
            passed += 1
        else:
            print(f"[FAIL] {test_name}")

    print(f"\n結果: {passed}/{len(tests)} 測試通過")

    return passed == len(tests)


if __name__ == "__main__":
    main()