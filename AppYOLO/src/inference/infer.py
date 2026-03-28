from .image import ImageInfer
from .video import VideoInfer


class YOLOInfer:
    def __init__(self, model_path="models/release.pt"):
        self.image_infer = ImageInfer(model_path)
        self.video_infer = VideoInfer(model_path)

    def run(self, path, save_path=None):
        if path.lower().endswith((".jpg", ".png", ".jpeg")):
            return self.image_infer.run(path)

        elif path.lower().endswith((".mp4", ".avi", ".mov")):
            return self.video_infer.run(path, save_path)

        else:
            raise ValueError("Unsupported file format")