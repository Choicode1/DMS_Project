
import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class EyeAnalyzer:

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(
        self,
        model_path,
        ear_threshold=0.20,
    ):
        self.left_ear_threshold = 0.18
        self.right_ear_threshold = 0.20

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )

        self.face_landmarker = (
            vision.FaceLandmarker.create_from_options(options)
        )

    @staticmethod
    def calculate_ear(landmarks, indices, width, height):

        points = np.array([
            [
                landmarks[idx].x * width,
                landmarks[idx].y * height
            ]
            for idx in indices
        ], dtype=np.float32)

        vertical_1 = np.linalg.norm(points[1] - points[5])
        vertical_2 = np.linalg.norm(points[2] - points[4])

        horizontal = np.linalg.norm(points[0] - points[3])

        if horizontal < 1e-6:
            return 0.0

        return float(
            (vertical_1 + vertical_2) / (2.0 * horizontal)
        )

    def analyze(self, frame):

        height, width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = self.face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]

        left_ear = self.calculate_ear(
            landmarks,
            self.LEFT_EYE,
            width,
            height
        )

        right_ear = self.calculate_ear(
            landmarks,
            self.RIGHT_EYE,
            width,
            height
        )

        avg_ear = (left_ear + right_ear) / 2.0

        left_closed = left_ear < self.left_ear_threshold
        right_closed = right_ear < self.right_ear_threshold

        eye_closed = left_closed and right_closed

        return {
            'left_ear': left_ear,
            'right_ear': right_ear,
            'avg_ear': avg_ear,
            'left_closed': left_closed,
            'right_closed': right_closed,
            'eye_closed': eye_closed,
            'landmarks': landmarks,
        }

    def draw(self, frame, result):

        if result is None:
            return frame

        output = frame.copy()

        height, width = output.shape[:2]
        landmarks = result['landmarks']

        for indices in [self.LEFT_EYE, self.RIGHT_EYE]:

            for idx in indices:

                landmark = landmarks[idx]

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    output,
                    (x, y),
                    2,
                    (0, 255, 255),
                    -1
                )

        status = (
            'CLOSED'
            if result['eye_closed']
            else 'OPEN'
        )

        color = (
            (0, 0, 255)
            if result['eye_closed']
            else (0, 255, 0)
        )

        cv2.putText(
            output,
            f"EAR: {result['avg_ear']:.3f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        cv2.putText(
            output,
            f"Eye: {status}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        return output

    def close(self):
        self.face_landmarker.close()
