import gymnasium as gym
from gymnasium import spaces
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class RedBall(Node):
    """
    ROS 2 node to detect red balls and store latest frame/detection flag.
    """
    def __init__(self):
        super().__init__('redball')
        self.subscription = self.create_subscription(
            Image,
            'custom_ns/camera1/image_raw',
            self.listener_callback,
            10)
        self.br = CvBridge()
        self.target_publisher = self.create_publisher(Image, 'target_redball', 10)
        self.twist_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        self.latest_frame = None
        self.ball_detected = False

    def listener_callback(self, msg):
        # Convert ROS image to OpenCV format
        frame = self.br.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.latest_frame = frame.copy()
        self.ball_detected = False

       
        hsv_conv_img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Correct red HSV range
        lower_red1 = (0, 100, 100)
        upper_red1 = (10, 255, 255)
        lower_red2 = (160, 100, 100)
        upper_red2 = (179, 255, 255)
        mask1 = cv2.inRange(hsv_conv_img, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_conv_img, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Blur and morph
        blurred_mask = cv2.GaussianBlur(red_mask, (9, 9), 3, 3)
        erode_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate_element = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        eroded_mask = cv2.erode(blurred_mask, erode_element)
        dilated_mask = cv2.dilate(eroded_mask, dilate_element)

        detected_circles = cv2.HoughCircles(
            dilated_mask, cv2.HOUGH_GRADIENT, 1, 150,
            param1=100, param2=20, minRadius=2, maxRadius=2000
        )

        if detected_circles is not None:
            self.get_logger().info('Red ball detected!')
            self.ball_detected = True
            for circle in detected_circles[0, :]:
                circled_orig = cv2.circle(
                    frame, (int(circle[0]), int(circle[1])),
                    int(circle[2]), (0, 255, 0), thickness=3
                )
            self.target_publisher.publish(self.br.cv2_to_imgmsg(circled_orig, encoding='rgb8'))
        else:
            self.get_logger().info('No ball detected')

class CreateRedBall(gym.Env):
    """
    Gymnasium environment wrapping the RedBall ROS2 node.
    Observations: latest camera frame
    Actions: 0 = left, 1 = right
    """
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None):
        super().__init__()
        rclpy.init()

        self.redball = RedBall()
        self.action_space = spaces.Discrete(2)  # left, right

        # Wait for first image to set observation space
        print("Waiting for first image from /custom_ns/camera1/image_raw...")
        while rclpy.ok() and self.redball.latest_frame is None:
            rclpy.spin_once(self.redball, timeout_sec=0.5)

        if self.redball.latest_frame is not None:
            h, w, c = self.redball.latest_frame.shape
            self.observation_space = spaces.Box(low=0, high=255, shape=(h, w, c), dtype=np.uint8)
            print(f"Image shape detected: {h}x{w}x{c}")
        else:
            print("No image received; defaulting to 480x640x3")
            self.observation_space = spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)

        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        rclpy.spin_once(self.redball, timeout_sec=0.5)
        obs = self.redball.latest_frame if self.redball.latest_frame is not None else np.zeros(self.observation_space.shape, dtype=np.uint8)
        return obs, {}

    def step(self, action):
        twist = Twist()
        if action == 0:
            twist.angular.z = 0.5  
        elif action == 1:
            twist.angular.z = -0.5  
        self.redball.twist_publisher.publish(twist)

        rclpy.spin_once(self.redball, timeout_sec=0.2)

        obs = self.redball.latest_frame if self.redball.latest_frame is not None else np.zeros(self.observation_space.shape, dtype=np.uint8)
        reward = 1.0 if self.redball.ball_detected else 0.0
        terminated = False
        truncated = self.step_count >= 100
        self.step_count += 1

        return obs, reward, terminated, truncated, {"red_ball_detected": self.redball.ball_detected}

    def close(self):
        self.redball.destroy_node()
        rclpy.shutdown()
