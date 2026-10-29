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

from dms_core.face_detector import FaceDetector



class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        default_model = os.path.join(
            get_package_share_directory('dms_core'),
            'models',
            'face_detection_yunet_2023mar.onnx',
        )

        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('model_path', default_model)
        self.declare_parameter('score_threshold', 0.8)

        image_topic = self.get_parameter('image_topic').value
        model_path = self.get_parameter('model_path').value
        score_threshold = float(
            self.get_parameter('score_threshold').value
        )

        self.bridge = CvBridge()
        self.face_detector = FaceDetector(
            model_path=model_path,
            score_threshold=score_threshold,
        )

        self.debug_publisher = self.create_publisher(
            Image,
            '/dms/face_debug_image',
            10,
        )

        self.frame_count = 0
        self.frames_since_log = 0
        self.last_fps_time = time.monotonic()
        self.last_error_time = 0.0

        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(f'Perception node started. Waiting for {image_topic}...')
        self.get_logger().info(f'YuNet Model Loaded: {model_path}')
        

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8',
            )

            faces = self.face_detector.detect(frame)

            for face in faces:
                x, y, width, height = map(
                    int, face[:4]
                )
                confidence = float(face[14])

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f'Face: {confidence:.2f}',
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

                landmarks = face[4:14].reshape(5, 2)

                for point_x, point_y in landmarks:
                    cv2.circle(
                        frame,
                        (int(point_x), int(point_y)),
                        2,
                        (0, 0, 255),
                        -1,
                    )

            debug_msg = self.bridge.cv2_to_imgmsg(
                frame,
                encoding='bgr8',
            )
            debug_msg.header = msg.header
            self.debug_publisher.publish(debug_msg)

            self.frame_count += 1
            self.frames_since_log += 1

            now = time.monotonic()
            elapsed = now - self.last_fps_time

            if elapsed >= 5.0:
                fps = self.frames_since_log / elapsed

                self.get_logger().info(
                    f'Input processed FPS: {fps:.1f}, '
                    f'faces detected: {len(faces)}'
                )

                self.frames_since_log = 0
                self.last_fps_time = now

        except Exception as exc:
            now = time.monotonic()

            if now - self.last_error_time >= 5.0:
                self.get_logger().error(
                    f'Perception error: {exc}'
                )
                self.last_error_time = now


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
