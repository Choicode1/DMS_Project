# 영상 구독 및 인식 결과 발행
import os
import time

import cv2
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from dms_msgs.msg import DriverState

from dms_core.face_detector import FaceDetector
from dms_core.eye_analyzer import EyeAnalyzer
from dms_core.perclos_tracker import PerclosTracker


class PerceptionNode(Node):

    def __init__(self):
        super().__init__('perception_node')

        package_path = get_package_share_directory('dms_core')

        default_yunet_model = os.path.join(
            package_path,
            'models',
            'face_detection_yunet_2023mar.onnx'
        )

        default_landmarker_model = os.path.join(
            package_path,
            'models',
            'face_landmarker.task'
        )

        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('yunet_model_path', default_yunet_model)
        self.declare_parameter('landmarker_model_path', default_landmarker_model)
        self.declare_parameter('score_threshold', 0.8)
        self.declare_parameter('ear_threshold', 0.20)
        self.declare_parameter('perclos_window_sec', 60.0)

        image_topic = self.get_parameter('image_topic').value
        yunet_model = self.get_parameter('yunet_model_path').value
        landmarker_model = self.get_parameter('landmarker_model_path').value

        score_threshold = float(self.get_parameter('score_threshold').value)
        ear_threshold = float(self.get_parameter('ear_threshold').value)

        perclos_window = float(self.get_parameter('perclos_window_sec').value)

        self.bridge = CvBridge()

        self.eye_closed_start = None
        self.eye_closed_duration = 0.0

        # 1. YuNet 얼굴 검출기
        self.face_detector = FaceDetector(
            model_path=yunet_model,
            score_threshold=score_threshold
        )

        # 2. MediaPipe 눈 분석기
        self.eye_analyzer = EyeAnalyzer(
            model_path=landmarker_model,
            ear_threshold=ear_threshold
        )

        # 3. PERCLOS 계산기
        self.perclos_tracker = PerclosTracker(
            window_sec=perclos_window,
            max_gap_sec=0.5,
            min_observed_sec=10.0
        )

        # 디버그 영상
        self.debug_publisher = self.create_publisher(
            Image,
            '/dms/face_debug_image',
            10
        )

        # 운전자 상태 메시지
        self.state_publisher = self.create_publisher(
            DriverState,
            '/dms/driver_state',
            10
        )

        # 카메라 구독
        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.frame_count = 0
        self.frames_since_log = 0
        self.last_fps_time = time.monotonic()
        self.last_error_time = 0.0

        self.get_logger().info(
            f'Perception node started: {image_topic}'
        )

        self.get_logger().info(
            f'YuNet loaded: {yunet_model}'
        )

        self.get_logger().info(
            f'Face Landmarker loaded: {landmarker_model}'
        )

    def image_callback(self, msg: Image):

        now = time.monotonic()

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

            # --------------------------------
            # 1. YuNet 얼굴 검출
            # --------------------------------
            faces = self.face_detector.detect(frame)

            face_detected = len(faces) > 0

            for face in faces:

                x, y, width, height = map(
                    int,
                    face[:4]
                )

                confidence = float(face[14])

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f'Face: {confidence:.2f}',
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

            # --------------------------------
            # 2. MediaPipe 눈 분석
            # --------------------------------
            eye_result = self.eye_analyzer.analyze(frame)

            eye_state_valid = eye_result is not None

            # --------------------------------
            # 3. PERCLOS 계산
            # --------------------------------
            if eye_state_valid:

                left_ear = eye_result['left_ear']
                right_ear = eye_result['right_ear']
                eye_closed = eye_result['eye_closed']

                perclos_result = self.perclos_tracker.update(
                    eye_closed,
                    now
                )

                # 눈 랜드마크 및 EAR 시각화
                frame = self.eye_analyzer.draw(
                    frame,
                    eye_result
                )

            else:
                left_ear = 0.0
                right_ear = 0.0
                eye_closed = False

                # 얼굴 미검출은 눈 감김으로 계산하지 않음
                perclos_result = self.perclos_tracker.update(
                    None,
                    now
                )

            # --------------------------------
            # 3-1. 눈 감김 지속시간 계산
            # --------------------------------
            if eye_state_valid and eye_closed:

                if self.eye_closed_start is None:
                    self.eye_closed_start = now

                self.eye_closed_duration = (
                    now - self.eye_closed_start
                )

            else:
                self.eye_closed_start = None
                self.eye_closed_duration = 0.0

            # --------------------------------
            # 4. DriverState 메시지 생성
            # --------------------------------
            state = DriverState()

            state.header = msg.header
            state.camera_ok = True
            state.face_detected = face_detected or eye_state_valid

            state.eye_state_valid = eye_state_valid
            state.eye_closed = eye_closed

            state.left_ear = float(left_ear)
            state.right_ear = float(right_ear)

            state.eye_closed_duration = float(self.eye_closed_duration)

            state.perclos_valid = perclos_result['ready']
            state.perclos = (
                float(perclos_result['perclos'])
                if perclos_result['ready']
                else 0.0
            )

            # 아직 구현하지 않은 기능
            state.head_yaw_deg = 0.0
            state.head_pitch_deg = 0.0
            state.gaze_away = False

            self.state_publisher.publish(state)

            # --------------------------------
            # 5. 디버그 영상 발행
            # --------------------------------
            if perclos_result['ready']:
                perclos_text = (
                    f"PERCLOS: "
                    f"{perclos_result['perclos_percent']:.1f}%"
                )
            else:
                perclos_text = "PERCLOS: WARMING UP"

            cv2.putText(
                frame,
                perclos_text,
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            debug_msg = self.bridge.cv2_to_imgmsg(
                frame,
                encoding='bgr8'
            )

            debug_msg.header = msg.header

            self.debug_publisher.publish(debug_msg)

            # --------------------------------
            # 6. FPS 및 상태 로그
            # --------------------------------
            self.frame_count += 1
            self.frames_since_log += 1

            elapsed = now - self.last_fps_time

            if elapsed >= 5.0:

                fps = self.frames_since_log / elapsed

                self.get_logger().info(
                    f'FPS: {fps:.1f} | '
                    f'Face: {face_detected} | '
                    f'Eye valid: {eye_state_valid} | '
                    f'Eye closed: {eye_closed} | '
                    f'PERCLOS: {perclos_result["perclos_percent"]:.1f}%'
                )

                self.frames_since_log = 0
                self.last_fps_time = now

        except Exception as exc:

            if now - self.last_error_time >= 5.0:

                self.get_logger().error(
                    f'Perception error: {exc}'
                )

                self.last_error_time = now

    def destroy_node(self):

        self.eye_analyzer.close()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = None

    try:
        node = PerceptionNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as exc:
        if node is not None:
            node.get_logger().fatal(
                f'Perception node failed: {exc}'
            )
        else:
            print(f'Perception initialization failed: {exc}')

        raise

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
