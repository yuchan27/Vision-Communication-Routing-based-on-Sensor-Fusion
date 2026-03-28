import os
from pathlib import Path
from ultralytics import YOLO
import cv2

from .utils import convert_to_yolo_format


class ImageInfer:
    def __init__(self, model_path="models/release.pt"):
        self.model = YOLO(model_path)
        self.output_dir = Path("outputs/images")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, image_path, save=True):
        results = self.model(image_path, verbose=False)
        result = results[0]

        # 轉 YOLO 格式輸出
        yolo_results = convert_to_yolo_format(result)

        # 存圖片
        if save:
            annotated = result.plot(conf=False)  # 不顯示 confidence
            filename = os.path.basename(image_path)
            save_path = self.output_dir / filename
            cv2.imwrite(str(save_path), annotated)

        return yolo_results