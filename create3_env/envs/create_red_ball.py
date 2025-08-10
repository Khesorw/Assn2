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
    def __init__(self):
        super().__init__('redball')
        self.subscription = self.create_subscription(
            Image,
            'custom_ns/camera1/image_raw',
            self.listener_callback,
            10
        )
        self.br = CvBridge()
        self.twist_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        self.latest_frame = None
        self.ball_detected = False
        self.redball_x = -1  # -1 means "not detected"

    def listener_callback(self, msg):
        frame = self.br.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.latest_frame = frame.copy()
        self.ball_detected = False
        self.redball_x = -1

        hsv_img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        lower_red1 = (0, 100, 100)
        upper_red1 = (10, 255, 255)
        lower_red2 = (160, 100, 100)
        upper_red2 = (179, 255, 255)
        mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        blurred_mask = cv2.GaussianBlur(red_mask, (9, 9), 3, 3)
        eroded = cv2.erode(blurred_mask, np.ones((3, 3), np.uint8))
        dilated = cv2.dilate(eroded, np.ones((8, 8), np.uint8))

        circles = cv2.HoughCircles(
            dilated, cv2.HOUGH_GRADIENT, 1, 150,
            param1=100, param2=20, minRadius=2, maxRadius=2000
        )

        if circles is not None:
            self.ball_detected = True
            x = int(circles[0, 0, 0])
            self.redball_x = max(0, min(x, 640))  # clamp to [0, 640]
        else:
            self.redball_x = -1



class CreateRedBall(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        rclpy.init()
        self.redball = RedBall()

        self.action_space = spaces.Discrete(2)  # 0=left, 1=right
        # Observation is normalized x position of ball or 1.0 if not detected
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        print("Waiting for first image...")
        while rclpy.ok() and self.redball.latest_frame is None:
            rclpy.spin_once(self.redball, timeout_sec=0.5)

        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        rclpy.spin_once(self.redball, timeout_sec=0.5)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        twist = Twist()
        if action == 0:
            twist.angular.z = 0.5
        elif action == 1:
            twist.angular.z = -0.5
        self.redball.twist_publisher.publish(twist)

        rclpy.spin_once(self.redball, timeout_sec=0.2)

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= 100

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        if self.redball.redball_x >= 0:
            # Normalize by max x (641)
            norm_x = self.redball.redball_x / 641.0
            return np.array([norm_x], dtype=np.float32)
        else:
            return np.array([1.0], dtype=np.float32)  # no ball detected

    def _compute_reward(self, obs):
        # Reward is higher the closer the ball is to center (0.5)
        if obs[0] == 1.0:  # no ball
            return -1.0
        return 1.0 - abs(obs[0] - 0.5) * 2  # closer to center (0.5) better

    def close(self):
        self.redball.destroy_node()
        rclpy.shutdown()

