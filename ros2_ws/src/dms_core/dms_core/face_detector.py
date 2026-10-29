"""OpenCV YuNet 기반 얼굴 검출 모듈."""
from pathlib import Path

import cv2


class FaceDetector:
    
    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.8,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ):
        model = Path(model_path)

        if not model.is_file():
            raise FileNotFoundError(
                f"YuNet model not found: {model}"
            )

        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError(
                "cv2.FaceDetectorYN is unavailable. "
                "Check the installed OpenCV version."
            )

        self.detector = cv2.FaceDetectorYN.create(
            model=str(model),
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )

    def detect(self, frame):
        """
        Args:
            frame: OpenCV BGR image.

        Returns:
            List of face detections.
            Each detection contains 15 values:
            bbox, 5 facial landmarks, confidence score.
        """
        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]

        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)

        if faces is None:
            return []

        return faces
