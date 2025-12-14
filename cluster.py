import os
import math
import cv2
import numpy as np
import rclpy
import tf2_ros
from ultralytics import YOLO
import sort
from rclpy.node import Node
from collections import deque
from sensor_msgs.msg import CompressedImage, LaserScan, Image
from nav_msgs.msg import Odometry, OccupancyGrid
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge, CvBridgeError
from sklearn.cluster import DBSCAN
from rclpy.qos import QoSProfile, ReliabilityPolicy  
from visualization_msgs.msg import Marker, MarkerArray
from collections import deque


class LidarCamBasePolarNode(Node):
    def __init__(self):
        #initialize the node
        super().__init__('lidar_cam_base_polar_node')
      
        self.get_logger().info("LidarCamBasePolarNode has been initialized.")

        # Topics
        self.camera_topic = '/kachaka/front_camera/image_raw/compressed'
        self.lidar_topic  = '/kachaka/lidar/scan'
        self.odometry_topic  = '/kachaka/odometry/odometry'
        self.tf_topic = '/tf'
        self.map_topic = '/map'

        