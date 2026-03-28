import cv2
from ultralytics import YOLO
from .utils import convert_to_yolo_format


class VideoInfer:
    def __init__(self, model_path="models/release.pt"):
        self.model = YOLO(model_path)

    def run(self, video_path, save_path=None):
        cap = cv2.VideoCapture(video_path)

        all_results = []  # 每一frame的結果

        writer = None

        if save_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, verbose=False)
            result = results[0]

            # 取得 YOLO 格式結果
            yolo_result = convert_to_yolo_format(result)

            all_results.append({
                "frame_id": frame_id,
                "detections": yolo_result
            })

            # 視覺化（不顯示 confidence）
            annotated = result.plot(conf=False)

            if writer:
                writer.write(annotated)

            frame_id += 1

        cap.release()
        if writer:
            writer.release()

        return all_results