#!/usr/bin/env python3

import os
import math
import cv2
import numpy as np
import rclpy
import tf2_ros
from ultralytics import YOLO
from sort.sort import Sort
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


# 初期設定
class LidarCamBasePolarNode(Node):
    def __init__(self):
        super().__init__('lidar_cam_base_polar_node')
        
        # 1. YOLO及びCvBridgeの設定
        self.yolo_model = YOLO('yolo11n.pt')
        self.bridge = CvBridge()
        # COCOデータセット基準 : person=0, chair=56
        self.target_classes = {'chair': 56, 'person': 0}
        self.conf_threshold = 0.3 
        
        # 2. Parameters
        # 解像度
        self.height = 720
        self.width  = 1280
        # Intrinsic Parameter
        self.camera_matrix = np.array([
            [507.19726398283353, 0.0, 627.8761942540922],
            [0.0, 503.5946424433988, 322.01074367172447],
            [0.0, 0.0, 1.0]
        ])
        # Distortion Parameter
        self.dist_coeffs = np.array([
            -0.2868624114347061, 0.11514521907472543,
            0.0008145218734235378, 5.778678686282746e-06,
            -0.02491904639602924
        ])    
        # Extrinsic Parameter    
        self.extrinsic_matrix = np.array([
            [-1.0, -4.44089210e-16, 4.93038066e-32, -1.19015908e-17],
            [9.86076132e-32, -2.22044605e-16, -1.0, -3.49000000e-02],
            [4.44089210e-16, -1.0, 2.22044605e-16, -5.36000000e-02],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # 3. トピック設定
        self.camera_topic = '/kachaka/front_camera/image_raw/compressed'
        self.lidar_topic  = '/kachaka/lidar/scan'
        self.odometry_topic  = '/kachaka/odometry/odometry'
        self.tf_topic = '/tf'
        self.map_topic = '/map'
        
        # 4. QOS設定
        qos_profile_tf = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        qos_profile_best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        # 5. Subscription生成
        self.create_subscription(CompressedImage, self.camera_topic, self.camera_callback, qos_profile_best_effort)
        self.create_subscription(LaserScan, self.lidar_topic, self.lidar_callback, qos_profile_best_effort)
        self.create_subscription(Odometry, self.odometry_topic, self.odometry_callback, qos_profile_best_effort)
        self.create_subscription(OccupancyGrid, self.map_topic, self.map_callback, qos_profile_tf)
        
        # 6. Buffer, TransformListener初期化
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.camera_msgs = [] 
        self.lidar_msgs = deque(maxlen=10)
        
        # 7. Odometey変数
        self.current_odometry = None      
        self.last_saved_odometry = None   
        self.odom_threshold = 0.00        
        
        # 8. TF変数
        self.saved_transform = None
        
        # 9. SORT初期化
        self.tracker = Sort()

        # 10. Timer
        self.create_timer(2.0, self.timer_callback)
        
        # 11. MarkerArray publisher (RViz視覚化)
        self.marker_pub = self.create_publisher(MarkerArray, "visualization_marker_array", 10)

        # 12. occupancy値
        self.occupancy_threshold = 50
        
        # 13. Image publisher
        self.image_pub = self.create_publisher(Image, '/kachaka/front_camera/uncompressed', 10)
        
        
    # Markerの色設定
    def get_color_for_label(self, label: str):
        if not label:
            return (0.5, 0.5, 0.5)
        if not hasattr(self, 'label_color_map'):
            self.label_color_map = {}
            self.next_color_index = 0
            self.palette = [
                (1.0, 0.0, 1.0),  # magenta
                (0.0, 1.0, 1.0),  # cyan       
                (0.0, 1.0, 0.0),  # green
                (0.0, 0.0, 1.0),  # blue
                (1.0, 1.0, 0.0),  # yellow
            ]
        if label not in self.label_color_map:
            self.label_color_map[label] = self.palette[self.next_color_index % len(self.palette)]
            self.next_color_index += 1
        return self.label_color_map[label]


    # 新しい領域(new_region)が、既存に含まれる領域(semantic_assignments)と50%以上重なっている場合は、既存のラベルを返し、そうでなければ new_label を返す
    def merge_region(self, new_label, new_region, semantic_assignments, threshold=0.5):
        for existing_label, existing_region in semantic_assignments.items():
            if len(new_region) > 0 and (len(new_region.intersection(existing_region)) / len(new_region)) >= threshold:
                return existing_label
        return new_label
    
    
    # /mapメッセージCall back : OccupancyGrid情報をアップデート(原点位置, 格子の数, Orientationなど)
    def map_callback(self, msg):
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        # Orientationのyaw計算
        q = msg.info.origin.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
        grid_data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        self.map_info = {
            'width': width, 
            'height': height,
            'resolution': resolution,
            'origin_x': origin_x,
            'origin_y': origin_y,
            'yaw': yaw,
            'grid_data': grid_data
        }
        self.get_logger().info("Map updated.")
    
    
    # (x, y)座標を/map基準のgrid cell (i, j)に変換
    def cell_index(self, x, y):
        if not hasattr(self, 'map_info'):
            self.get_logger().warn("Map info not available yet.")
            return None
        origin_x = self.map_info['origin_x']
        origin_y = self.map_info['origin_y']
        resolution = self.map_info['resolution']
        yaw = self.map_info.get('yaw', 0)
        # 原点基準に移動
        dx = x - origin_x
        dy = y - origin_y
        # mapのorientationを考慮して-yawだけ回転
        cos_theta = math.cos(-yaw)
        sin_theta = math.sin(-yaw)
        rotated_x = cos_theta * dx - sin_theta * dy
        rotated_y = sin_theta * dx + cos_theta * dy
        i = int(math.floor(rotated_x / resolution))
        j = int(math.floor(rotated_y / resolution))
        return (i, j)
     
     
    # 2秒ごとにconf_grid.txtを読み込み、/map（OccupancyGrid）を参照して、conf_grid.txt内の各シードから8方向にBFS検索を実行し、その結果をsemantic_grid.txtファイルに保存し、MarkerArrayを生成してpub
    def timer_callback(self):
        # ファイルの経路設定
        detections_file = "/home/lim/output/lidar_detections.txt"
        grid_file = "/home/lim/output/lidar_grid.txt"
        conf_file = "/home/lim/output/conf_grid.txt"
        
        # 1. lidar_detections.txtを読み込み、各行ごとに/map基準のセル (i,j) を計算
        try:
            with open(detections_file, "r") as f:
                lines = f.readlines()
        except Exception as e:
            self.get_logger().error(f"Failed to read lidar detections file: {e}")
            return
        updated_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 6:
                self.get_logger().warn(f"Unexpected format in line: {line}")
                continue            
            sec = parts[0]
            nsec = parts[1]
            cls = parts[2]
            conf = parts[3]
            x_map_str = parts[4]
            y_map_str = parts[5]
            try:
                x_map = float(x_map_str)
                y_map = float(y_map_str)
            except Exception as e:
                self.get_logger().warn(f"Failed to convert coordinates in line: {line}")
                continue            
            cell = self.cell_index(x_map, y_map)
            if cell is None:
                continue
            i, j = cell
            updated_line = f"{sec},{nsec},{cls},{conf},{i},{j}\n"
            updated_lines.append(updated_line)        
        try:
            with open(grid_file, "w") as f:
                f.writelines(updated_lines)
            self.get_logger().info(f"Detection grid updated with {len(updated_lines)} entries.")
        except Exception as e:
            self.get_logger().error(f"Failed to write detection grid: {e}")
        
        # 2. lidar_grid.txtを読み込み、ID&セルごとにYOLO信頼度の合計を計算
        try:
            with open(grid_file, "r") as f:
                grid_lines = f.readlines()
        except Exception as e:
            self.get_logger().error(f"Failed to read grid file: {e}")
            return
        groups = {}
        for line in grid_lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 6:
                continue
            label = parts[2]  
            try:
                conf_val = float(parts[3])
                i = int(parts[4])
                j = int(parts[5])
            except Exception as e:
                continue
            if label not in groups:
                groups[label] = {}
            key = (i, j)
            groups[label][key] = groups[label].get(key, 0.0) + conf_val
        
        # 3. 各IDごとに累積信頼度が最も大きいセル(シード)を選択（合計が5を超えなければ保存しない）
        conf_lines = []
        for label, cell_dict in groups.items():
            if not cell_dict:
                continue
            sorted_cells = sorted(cell_dict.items(), key=lambda x: x[1], reverse=True)
            chosen = None
            for (cell, conf_sum) in sorted_cells:
                i, j = cell

                if self.map_info['grid_data'][j, i] >= self.occupancy_threshold:
                    chosen = (cell, conf_sum)
                    break
            if chosen is not None:
                (i, j), best_conf = chosen
                if best_conf <= 5:
                    continue
                conf_line = f"{label},{best_conf:.2f},{i},{j}\n"
                conf_lines.append(conf_line)
        
        # 4. conf_grid.txtに記録
        try:
            with open(conf_file, "w") as f:
                f.writelines(conf_lines)
            self.get_logger().info(f"Confidence grid updated with {len(conf_lines)} entries.")
        except Exception as e:
            self.get_logger().error(f"Failed to write confidence grid: {e}")
            
        # 5. conf_grid.txt作成後、semantic marker関数を呼び出す
        self.publish_semantic_markers()
        
        
    # Marker生成&pub
    def publish_semantic_markers(self):
        # ファイルの経路設定
        conf_file = "/home/lim/output/conf_grid.txt"
        semantic_file = "/home/lim/output/semantic_grid.txt"
        
        # 1. conf_grid.txt読み込み
        try:
            with open(conf_file, "r") as f:
                conf_lines = f.readlines()
        except Exception as e:
            self.get_logger().error(f"Failed to read conf grid file: {e}")
            return
        if not conf_lines:
            self.get_logger().warn("No conf_lines found; skipping semantic marker publish.")
            return
        
        # 2. /mapのOccupancyGrid情報を利用 (self.map_infoに最新/mapが保存)
        if self.map_info is None or 'grid_data' not in self.map_info:
            self.get_logger().warn("Map info or grid_data not available yet.")
            return
        occupancy_grid = self.map_info['grid_data']
        height = self.map_info['height']
        width = self.map_info['width'] 
        
        # 3. conf_grid.txtの各シードから8方向BFS検索実行
        semantic_assignments = {}  # { label: set((i, j), ...) }
        for line in conf_lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 4:
                self.get_logger().warn(f"Unexpected conf grid line: {line}")
                continue
            label = parts[0]
            try:
                seed_i = int(parts[2])
                seed_j = int(parts[3])
            except Exception as e:
                self.get_logger().warn(f"Failed to parse conf grid line: {line}")
                continue
            if seed_i < 0 or seed_i >= width or seed_j < 0 or seed_j >= height:
                continue
            if occupancy_grid[seed_j, seed_i] < self.occupancy_threshold :
                continue         
            region = set()
            queue = deque()
            queue.append((seed_i, seed_j))
            region.add((seed_i, seed_j))
            directions = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
            while queue:
                cx, cy = queue.popleft()
                for dx, dy in directions:
                    nx = cx + dx
                    ny = cy + dy
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    if (nx, ny) in region:
                        continue
                    if occupancy_grid[ny, nx] >= self.occupancy_threshold :
                        region.add((nx, ny))
                        queue.append((nx, ny))            
            # merge_regionの検索
            merged_label = self.merge_region(label, region, semantic_assignments, threshold=0.5)
            if merged_label in semantic_assignments:
                semantic_assignments[merged_label] = semantic_assignments[merged_label].union(region)
            else:
                semantic_assignments[merged_label] = region
        
        # 4. semantic_grid.txtに保存
        semantic_lines = []
        for label, cells in semantic_assignments.items():
            for (i, j) in cells:
                semantic_lines.append(f"{label},{i},{j}\n")
        try:
            with open(semantic_file, "w") as f:
                f.writelines(semantic_lines)
            self.get_logger().info(f"Semantic grid updated with {len(semantic_lines)} cells.")
        except Exception as e:
            self.get_logger().error(f"Failed to write semantic grid: {e}")
        
        # 5. Cube Marker & Text Marker pub
        markers = MarkerArray()
        marker_id = 0
        origin_x = self.map_info['origin_x']
        origin_y = self.map_info['origin_y']
        resolution = self.map_info['resolution']        
        for label, cells in semantic_assignments.items():
            r, g, b = self.get_color_for_label(label)
            sum_i = sum(cell[0] for cell in cells)
            sum_j = sum(cell[1] for cell in cells)
            count = len(cells)
            centroid_i = sum_i / count
            centroid_j = sum_j / count
            centroid_x = origin_x + (centroid_i + 0.5) * resolution
            centroid_y = origin_y + (centroid_j + 0.5) * resolution            
            # Cube Marker 
            for (i, j) in cells:
                x_cell = origin_x + (i + 0.5) * resolution
                y_cell = origin_y + (j + 0.5) * resolution                
                cube = Marker()
                cube.header.frame_id = "map"
                cube.header.stamp = self.get_clock().now().to_msg()
                cube.ns = "semantic_cells"
                cube.id = marker_id
                marker_id += 1
                cube.type = Marker.CUBE
                cube.action = Marker.ADD
                cube.pose.position.x = x_cell
                cube.pose.position.y = y_cell
                cube.pose.position.z = 0.0
                cube.pose.orientation.w = 1.0
                cube.scale.x = resolution * 0.9
                cube.scale.y = resolution * 0.9
                cube.scale.z = 0.1
                cube.color.r = r
                cube.color.g = g
                cube.color.b = b
                cube.color.a = 1.0
                cube.lifetime.sec = 1
                cube.lifetime.nanosec = 0
                markers.markers.append(cube)
            # Text Marker
            text = Marker()
            text.header.frame_id = "map"
            text.header.stamp = self.get_clock().now().to_msg()
            text.ns = "semantic_text"
            text.id = marker_id
            marker_id += 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = centroid_x
            text.pose.position.y = centroid_y
            text.pose.position.z = 0.2
            text.pose.orientation.w = 1.0
            text.scale.z = resolution * 0.5
            text.color.r = r
            text.color.g = g
            text.color.b = b
            text.color.a = 1.0
            text.text = label
            text.lifetime.sec = 1
            text.lifetime.nanosec = 0
            markers.markers.append(text)        
        self.marker_pub.publish(markers)
        self.get_logger().info(f"Published {len(markers.markers)} semantic markers.")
 
    
    # カメラコールバック関数
    def camera_callback(self, msg):
        self.get_logger().info("Camera message received")
        try:
            # 画像の圧縮解凍
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return       
        
        # img pub(RVIZ 視覚化)
        uncompressed_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.image_pub.publish(uncompressed_msg)
        
        # YOLO 検出
        results = self.yolo_model.track(cv_image, persist=True, classes=[0, 56])
        bboxes = []
        if len(results) > 0:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf >= self.conf_threshold:
                    track_id = int(box.id[0]) if box.id is not None else -1
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls_id = int(box.cls[0])
                    label = [name for name, cid in self.target_classes.items() if cid == cls_id][0]
                    combined_label = f"id:{track_id} {label}" if track_id >= 0 else label
                    bboxes.append((x1, y1, x2, y2, combined_label, conf))
        if bboxes:
            self.get_logger().info("Valid tracking box created")
            self.process_one_frame_with_boxes(msg, bboxes)
        ts = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        self.camera_msgs.append((msg, ts))


    # LiDAR コールバック関数
    def lidar_callback(self, msg):
        self.get_logger().info("LiDAR message received")
        ts = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        point_cloud = np.stack([x, y], axis=1)
        self.lidar_msgs.append((msg, ts, point_cloud))
    
    
    # Odometry コールバック関数
    def odometry_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.current_odometry = (x, y)
        self.get_logger().info(f"Odometry updated: x={x:.2f}, y={y:.2f}")
    
    
    # バウンディングボックス内のLiDAR点群のマッチング処理
    def process_one_frame_with_boxes(self, cam_msg, bounding_boxes):
        try:
            np_arr = np.frombuffer(cam_msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                return
            undist = cv2.undistort(img, self.camera_matrix, self.dist_coeffs)
            if not self.lidar_msgs:
                self.get_logger().warn("No LiDAR message")
                return
            lidar_msg, ts, point_cloud = self.lidar_msgs[-1]
            self.lidar_process_and_save(lidar_msg, bounding_boxes,
                                        cam_msg.header.stamp.sec, cam_msg.header.stamp.nanosec)
        except Exception as e:
            self.get_logger().error(f"Error in process_one_frame_with_boxes: {e}")

    
    # LiDAR データ処理&保存
    def lidar_process_and_save(self, lidar_msg, bounding_boxes, sec, nsec):
        ranges = np.array(lidar_msg.ranges)
        angles = np.linspace(lidar_msg.angle_min, lidar_msg.angle_max, len(ranges))        
        lidar_4d = np.vstack([
            ranges * np.cos(angles),
            ranges * np.sin(angles),
            np.zeros_like(ranges),
            np.ones_like(ranges)
        ])
        cam_4d = self.extrinsic_matrix @ lidar_4d
        z = cam_4d[2, :]
        front = (z > 0)
        if not np.any(front):
            return        
        cam_4d = cam_4d[:, front]
        uv = self.camera_matrix @ cam_4d[:3, :]
        uv /= cam_4d[2, :]
        pts_2d = uv[:2, :].T        
        valid_idx = np.where(front)[0]
        matched_pts_polar = []
        for i, (px, py) in enumerate(pts_2d):
            ix = int(px)
            iy = int(py)
            if 0 <= ix < self.width and 0 <= iy < self.height:
                boxes_found = []
                for (x1, y1, x2, y2, combined_label, cconf) in bounding_boxes:
                    if x1 <= ix <= x2 and y1 <= iy <= y2:
                        boxes_found.append((combined_label, cconf))
                if len(boxes_found) == 1:
                    clabel, cconf = boxes_found[0]
                    idx = valid_idx[i]
                    r = ranges[idx]
                    th = angles[idx]
                    matched_pts_polar.append([r, th, clabel, cconf])        
        if matched_pts_polar:
            self.get_logger().info(f"Matched LiDAR point cloud count: {len(matched_pts_polar)}")
            
            # (A) DBSCANフィルタリング
            matched_xy = []
            for (r, th, clabel, cconf) in matched_pts_polar:
                x = r * math.cos(th)
                y = r * math.sin(th)
                matched_xy.append([x, y])
            matched_xy = np.array(matched_xy)            
            db = DBSCAN(eps=0.1, min_samples=10).fit(matched_xy)
            labels = db.labels_
            cluster_idx = np.where(labels != -1)[0]
            if len(cluster_idx) == 0:
                self.get_logger().info("No cluster")
                return            
            filtered_matched_pts = [matched_pts_polar[i] for i in cluster_idx]
            
            # (B) オドメトリ変化確認（移動しなければ保存しない）
            if self.current_odometry is not None:
                if self.last_saved_odometry is not None:
                    dx = self.current_odometry[0] - self.last_saved_odometry[0]
                    dy = self.current_odometry[1] - self.last_saved_odometry[1]
                    distance = math.sqrt(dx**2 + dy**2)
                    if distance < self.odom_threshold:
                        self.get_logger().info("No odometry change")
                        return
                self.last_saved_odometry = self.current_odometry
            
            # (C) 点群をmap座標に変換後、lidar_detections.txtファイルに保存
            timestamp = f"{sec},{nsec}"
            detections_file = "/home/lim/output/lidar_detections.txt"
            try:
                with open(detections_file, "a") as f:
                    for pt in filtered_matched_pts:
                        r, th, clabel, cconf = pt
                        # 信頼度フィルタリング
                        if cconf < self.conf_threshold:
                            continue                       
                        new_range = r * math.sin(th + math.pi) + 0.156
                        new_angle = -r * math.cos(th + math.pi)
                        x_robot = new_range * math.cos(new_angle)
                        y_robot = new_range * math.sin(new_angle)                        
                        try:
                            transform = self.tf_buffer.lookup_transform("map", "base_footprint", rclpy.time.Time())
                        except Exception as e:
                            self.get_logger().warn(f"TF lookup failure: {e}")
                            continue                       
                        x_map, y_map = self.transform_point(x_robot, y_robot, transform)
                        line = f"{timestamp},{clabel},{cconf:.2f},{x_map:.3f},{y_map:.3f}\n"
                        f.write(line)
                self.get_logger().info("Lidar detections appended.")
            except Exception as e:
                self.get_logger().error(f"Failed to write lidar detections: {e}")
        else:
            self.get_logger().info("No LiDAR point clouds within the bounding box")
   
    
    # TF 変換(base_footprint -> map)
    def transform_point(self, x, y, transform):
        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        q = transform.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        x_map = math.cos(yaw) * x - math.sin(yaw) * y + tx
        y_map = math.sin(yaw) * x + math.cos(yaw) * y + ty
        return x_map, y_map
  
    
    # SORTトラッキング
    def process_tracking(self, detections, image):
        if len(detections) == 0:
            return
        dets = np.array([
            [det['bbox'][0], det['bbox'][1], det['bbox'][2], det['bbox'][3], det['confidence']]
            for det in detections
        ])
        tracked_objects = self.tracker.update(dets)
        
        def compute_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            return interArea / float(boxAArea + boxBArea - interArea)
        
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            track_box = [int(x1), int(y1), int(x2), int(y2)]
            track_id = int(track_id)
            best_iou = 0.0
            best_label = None
            for det in detections: 
                det_box = det['bbox']
                iou = compute_iou(track_box, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_label = det['label']
            combined_label = f"{best_label}{track_id}" if best_label is not None else f"unknown{track_id}"
            self.get_logger().info(f"Tracking box created: {combined_label}")
        self.get_logger().info("Tracking result update complete")
    
    
def main(args=None):
    rclpy.init(args=args)
    node = LidarCamBasePolarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
