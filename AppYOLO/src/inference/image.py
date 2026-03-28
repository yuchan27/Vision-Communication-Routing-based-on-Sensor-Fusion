import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime, timezone

from .utils import convert_to_yolo_format
from ..decision_engine import SafetyDecisionEngine


class ImageInfer:
    def __init__(self, model_path="models/release.pt"):
        self.model = YOLO(model_path)
        self.output_dir = Path("outputs/images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 整合 SafetyDecisionEngine
        self.decision_engine = SafetyDecisionEngine(fps=30, alarm_threshold=0.75)

    def run(self, image_path, save=True):
        """原有功能（完全不變）"""
        results = self.model(image_path, verbose=False)
        result = results[0]

        yolo_results = convert_to_yolo_format(result)

        if save:
            annotated = result.plot(conf=False)
            filename = os.path.basename(image_path)
            save_path = self.output_dir / filename
            cv2.imwrite(str(save_path), annotated)

        return yolo_results

    def _estimate_temperature_from_bbox(self, image_path, yolo_result):
        """
        科學熱力學公式（雙色高溫計）估測火焰溫度
        已修正為 class_id == 1（你的模型實際的 Fire 類別）
        """
        img = cv2.imread(image_path)
        if img is None:
            return None

        # 🔥 修正：使用 class_id = 1（你的模型 Fire 的真實 ID）
        FIRE_CLASS_ID = 1

        fire_boxes = []
        for box in yolo_result.boxes:
            if int(box.cls[0]) == FIRE_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                fire_boxes.append((x1, y1, x2, y2))

        if not fire_boxes:
            return None

        # 使用所有 fire box 的平均溫度（更穩定）
        temps = []
        for x1, y1, x2, y2 in fire_boxes:
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            avg_r = float(np.mean(crop[:, :, 2]))
            avg_g = float(np.mean(crop[:, :, 1]))
            if avg_g < 1.0:
                avg_g = 1.0

            lambda_r = 700e-9
            lambda_g = 546.1e-9
            C2 = 0.014388
            Cg = 1.0

            ratio = avg_r / avg_g
            lambda_ratio5 = (lambda_r / lambda_g) ** 5
            arg = (1.0 / Cg) * ratio * lambda_ratio5

            if arg <= 0:
                continue

            ln_arg = np.log(arg)
            temp_k = C2 * (1 / lambda_g - 1 / lambda_r) / ln_arg
            temp_k = max(300.0, min(3000.0, temp_k))
            temps.append(temp_k - 273.15)

        return float(np.mean(temps)) if temps else None

    def _to_yolo_format_str(self, result):
        """強制轉成 decision_engine 相容的 YOLO txt 字串（含 confidence）"""
        if len(result.boxes) == 0:
            return ""

        cls_list = result.boxes.cls.tolist()
        conf_list = result.boxes.conf.tolist()
        xywhn_list = result.boxes.xywhn.tolist()

        lines = []
        for cls_id, conf_val, (x, y, w, h) in zip(cls_list, conf_list, xywhn_list):
            line = f"{int(cls_id)} {x:.4f} {y:.4f} {w:.4f} {h:.4f} {conf_val:.4f}"
            lines.append(line)
        return "\n".join(lines)

    def run_with_decision(self, image_path, save=True, frame_id=0):
        """整合版執行流程（已修正）"""
        results = self.model(image_path, verbose=False)
        result = results[0]
        yolo_results = convert_to_yolo_format(result)   # 保留給使用者的原始格式

        if save:
            annotated = result.plot(conf=False)
            filename = os.path.basename(image_path)
            save_path = self.output_dir / filename
            cv2.imwrite(str(save_path), annotated)

        vision_temp_c = self._estimate_temperature_from_bbox(image_path, result)

        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        visual_objects_str = self._to_yolo_format_str(result)

        payload = {
            "context": {
                "timestamp": timestamp,
                "frame_id": frame_id
            },
            "perceptions": {
                "visual_objects": visual_objects_str,
                "environmental_sensors": {
                    "temperature_celsius": vision_temp_c if vision_temp_c is not None else 25.4
                }
            }
        }

        decision_result = self.decision_engine.evaluate_payload(payload)

        return yolo_results, decision_result, vision_temp_c