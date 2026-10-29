# 영상 구독 및 인식 결과 발행
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        self.bridge = CvBridge()
        self.frame_count = 0

        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            'Perception node started. Waiting for /image_raw...'
        )

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

            self.frame_count += 1

            if self.frame_count == 1:
                self.get_logger().info(
                    f'First frame received: '
                    f'{frame.shape[1]}x{frame.shape[0]}'
                )

        except Exception as exc:
            self.get_logger().error(
                f'Image conversion failed: {exc}'
            )


def main(args=None):
    rclpy.init(args=args)

    node = PerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
