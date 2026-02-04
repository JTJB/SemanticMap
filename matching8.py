#!/usr/bin/env python3

import os
import time
import math
import cv2
import numpy as np
import rclpy
import tf2_ros
from ultralytics import YOLO
import sort
from rclpy.node import Node
from collections import deque
from sensor_msgs.msg import CompressedImage, LaserScan, Image, PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy
from nav_msgs.msg import Odometry, OccupancyGrid
from tf2_msgs.msg import TFMessage
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge, CvBridgeError
from sklearn.cluster import DBSCAN
from rclpy.qos import QoSProfile, ReliabilityPolicy  
from visualization_msgs.msg import Marker, MarkerArray
from collections import deque
from collections import defaultdict
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import rclpy.time


# edited
def are_regions_adjacent(region1, region2, max_distance=2):
    """
    Check if two regions are close enough to merge.
    Returns True if any cell in region1 is within max_distance of any cell in region2.
    """
    for (i1, j1) in region1:
        for (i2, j2) in region2:
            # Chebyshev distance (max of absolute differences)
            distance = max(abs(i1 - i2), abs(j1 - j2))
            if distance <= max_distance:
                return True
    return False


# Union-Find structure for efficient merging
class UnionFind:
    def __init__(self, labels):
        self.parent = {label: label for label in labels}
    
    def find(self, label):
        if self.parent[label] != label:
            self.parent[label] = self.find(self.parent[label])  # Path compression
        return self.parent[label]
    
    def union(self, label1, label2):
        root1 = self.find(label1)
        root2 = self.find(label2)
        if root1 != root2:
            self.parent[root2] = root1


# 初期設定
class LidarCamBasePolarNode(Node):
    def __init__(self):
        super().__init__('lidar_cam_base_polar_node')

        folder_name = '/home/ting/output/folder_' + time.strftime("%Y_%m_%d_%H_%M_%S")
        os.mkdir(folder_name)
        self.folder=folder_name
        
        # 1. YOLO及びCvBridgeの設定
        self.yolo_model = YOLO('yolo11n.pt')
        self.bridge = CvBridge()
        # COCOデータセット基準 : person=0, chair=56, table=60
        self.target_classes = {'chair': 56, 'person': 0, 'table':60}
        self.conf_threshold = 0.1 #0.3
        
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
            [-1.0, 0.0, -7.34641021e-06, -1.96900963e-07],
            [7.34641021e-06, 3.67320510e-06, -1.0, -5.36000000e-02],
            [0.0, -1.0, -3.67320510e-06, -3.49000000e-02],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # self.extrinsic_matrix = np.array([
        #     [ 0.0, -1.0, -3.67320510e-06, -3.49000000e-02],
        #     [ 7.34641021e-06, 3.67320510e-06, -1.0, -5.36000000e-02],
        #     [ 1.0, 0.0, 7.34641021e-06, 1.96900963e-07],
        #     [ 0.0, 0.0, 0.0, 1.0]
        # ])

        
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
        self.hull_vertices = None   # Initialized to None to prevent AttributeError
        self.hull_polygon = None 
        
        # 7. Odometey変数
        self.current_odometry = None      
        self.last_saved_odometry = None   
        self.odom_threshold = 0.00        
        
        # 8. TF変数
        self.saved_transform = None
        
        # 9. SORT初期化
        self.tracker = sort.SORT()

        # 10. Timer
        self.create_timer(2.0, self.timer_callback)
        
        # 11. MarkerArray publisher (RViz視覚化)
        self.marker_pub = self.create_publisher(MarkerArray, "visualization_marker_array", 10)
        self.hull_viz_pub = self.create_publisher(MarkerArray, '/hull_visualization', 10)

        # 12. occupancy値
        self.occupancy_threshold =70
        
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

    # "id:3 chair" -> "chair 3"  (chair + id)
    def format_display_label(self, raw_label: str) -> str:
        parts = raw_label.split()
        if parts and parts[0].startswith("id:"):
            obj_id = parts[0].split(":", 1)[1]
            obj_name = " ".join(parts[1:]) if len(parts) > 1 else ""
            if obj_name and obj_id:
                return f"{obj_name} {obj_id}"
            if obj_name:
                return obj_name
        return raw_label


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
            'resolution': 0.1, #original 0.05
            'origin_x': origin_x,
            'origin_y': origin_y,
            'yaw': yaw,
            'grid_data': grid_data
        }
        self.get_logger().error(f"Map updated with resolution: {resolution:.2f}.")
    
    
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
     
        # grid cell (i, j) を /map の (x, y) に変換
    def cell_to_map(self, i, j):
        if self.map_info is None:
            self.get_logger().warn("Map info not available yet for cell_to_map.")
            return None
        origin_x = self.map_info['origin_x']
        origin_y = self.map_info['origin_y']
        resolution = self.map_info['resolution']
        yaw = self.map_info.get('yaw', 0)

        # 1. Cell 中心のローカル座標
        rotated_x = (i + 0.5) * resolution
        rotated_y = (j + 0.5) * resolution

        # 2. Map orientation を考慮した逆回転
        dx = math.cos(yaw) * rotated_x - math.sin(yaw) * rotated_y
        dy = math.sin(yaw) * rotated_x + math.cos(yaw) * rotated_y

        # 3. Origin を足して map 座標へ
        x = origin_x + dx
        y = origin_y + dy
        return (x, y)
    

    # 2秒ごとにconf_grid.txtを読み込み、/map（OccupancyGrid）を参照して、conf_grid.txt内の各シードから8方向にBFS検索を実行し、その結果をsemantic_grid.txtファイルに保存し、MarkerArrayを生成してpub
    
    #add the unknown lidar points
    #clustering alienate unknown points
    #use dbscan to remove background
    def timer_callback(self):
        # ファイルの経路設定
        detections_file = self.folder + "/lidar_detections.txt"
        unknowns_file = self.folder + "/lidar_unknowns.txt"
        grid_file = self.folder + "/lidar_grid.txt"
        conf_file = self.folder + "/conf_grid.txt"
        hull_file = self.folder + "/hull.txt"

        # 🔹 出力ディレクトリを自動作成
        output_dir = os.path.dirname(detections_file)
        os.makedirs(output_dir, exist_ok=True)

        # 🔹 ファイルが存在しない場合は作成
        for f in [detections_file, grid_file, conf_file]:
            if not os.path.exists(f):
                open(f, 'w').close()  # 空ファイルを作成
        
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

        

        # 4. Write merged results
        conf_lines = []
        for label, cells_dict in groups.items():
            for (i, j), confidence in cells_dict.items():
                conf_lines.append(f"{label},{confidence},{i},{j},\n")  # Include confidence in output

        # 4. conf_grid.txtに記録
        try:
            with open(conf_file, "w") as f:
                f.writelines(conf_lines)
        except Exception as e:
            self.get_logger().error(f"Failed to write confidence grid: {e}")
            
        # 5. conf_grid.txt作成後、semantic marker関数を呼び出す
        self.publish_semantic_markers()
        
        


     # Marker生成&pub
    def publish_semantic_markers(self):
        conf_file = self.folder + "/conf_grid.txt"
        semantic_file = self.folder + "/semantic_grid.txt"
        grid_file = self.folder + "/lidar_grid.txt"

        # # 1. conf_grid.txt読み込み
        # try:
        #     with open(conf_file, "r") as f:
        #         conf_lines = f.readlines()
        # except Exception as e:
        #     self.get_logger().error(f"Failed to read conf grid file: {e}")
        #     return
        # if not conf_lines:
        #     self.get_logger().warn("No conf_lines found; skipping semantic marker publish.")
        #     return

        # # 2. /mapのOccupancyGrid情報を利用
        # if self.map_info is None or 'grid_data' not in self.map_info:
        #     self.get_logger().warn("Map info or grid_data not available yet.")
        #     return
        # occupancy_grid = self.map_info['grid_data']
        # height = self.map_info['height']
        # width = self.map_info['width']

        # # 3. conf_grid.txtの各シードから8方向BFS検索実行
        semantic_assignments = {}  # { label: set((i, j), ...) }
        # for line in conf_lines:
        #     line = line.strip()
        #     if not line:
        #         continue
        #     parts = line.split(',')
        #     if len(parts) != 4:
        #         self.get_logger().warn(f"Unexpected conf grid line: {line}")
        #         continue
        #     label = parts[0]
        #     try:
        #         seed_i = int(parts[2])
        #         seed_j = int(parts[3])
        #     except Exception:
        #         self.get_logger().warn(f"Failed to parse conf grid line: {line}")
        #         continue
        #     if seed_i < 0 or seed_i >= width or seed_j < 0 or seed_j >= height:
        #         continue
        #     if occupancy_grid[seed_j, seed_i] < self.occupancy_threshold:
        #         continue

        #     region = set()
        #     queue = deque()
        #     queue.append((seed_i, seed_j))
        #     region.add((seed_i, seed_j))

        #     directions = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
        #     while queue and len(region) < 70:
        #         cx, cy = queue.popleft()
        #         for dx, dy in directions:
        #             nx = cx + dx
        #             ny = cy + dy

        #             if nx < 0 or nx >= width or ny < 0 or ny >= height:
        #                 continue
        #             if (nx, ny) in region:
        #                 continue
        #             # if occupancy_grid[ny, nx] >= self.occupancy_threshold:
        #             #     region.add((nx, ny))
        #             #     queue.append((nx, ny))


        #     # merge_regionの検索
        #     merged_label = self.merge_region(label, region, semantic_assignments, threshold=0.5)
        #     if merged_label in semantic_assignments:
        #         semantic_assignments[merged_label] = semantic_assignments[merged_label].union(region)
        #     else:
        #         semantic_assignments[merged_label] = region

        # 4. semantic_grid.txtに保存
        semantic_lines = []
        # for label, cells in semantic_assignments.items():
        #     for (i, j) in cells:
        #         semantic_lines.append(f"{label},{i},{j}\n")
        # Read the file and parse each line
        tmp = {}
        with open(conf_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                row = line.split(',')
                
                label = row[0].strip()
                confidence = float(row[1].strip())
                x = float(row[2].strip())
                y = float(row[3].strip())
                if label not in tmp:
                    tmp[label] = set()

                if confidence > 5:
                    tmp[label].add((x, y, confidence))
                semantic_lines.append(f"{row[0]},({row[2]},{row[3]})\n")
        try:
            with open(semantic_file, "w") as f:
                f.writelines(semantic_lines)
        except Exception as e:
            self.get_logger().error(f"Failed to write semantic grid: {e}")


        filtered_tmp = {}
        semantic_lines = []
        
        MARGIN_MULTIPLIER = 0.7  # Adjust this: higher = more aggressive filtering
        
        for label, points in tmp.items():
            if len(points) < 3:  # Skip labels with too few points
                continue
            
            # Extract confidences for this label
            confidences = np.array([conf for (x, y, conf) in points])
            
            # Calculate statistics
            mean_conf = np.mean(confidences)
            std_conf = np.std(confidences)
            median_conf = np.median(confidences)
            
            # === CHOOSE YOUR THRESHOLD METHOD ===
            
            # Option 1: Mean + margin (good for normal distributions)
            # threshold = mean_conf + MARGIN_MULTIPLIER * std_conf
            
            # Option 2: Percentile-based (keep top X%)
            threshold = np.percentile(confidences, 90)  # Keep top 10%
            
            # Option 3: Median + fixed margin
            # threshold = median_conf + 1.0
            
            # Option 4: Mean only (simpler)
            # threshold = mean_conf * 1.2
            
            self.get_logger().info(
                f"Label '{label}': mean={mean_conf:.2f}, std={std_conf:.2f}, "
                f"threshold={threshold:.2f}, points={len(points)}"
            )
            
            # Filter points above threshold
            if label not in filtered_tmp:
                filtered_tmp[label] = {}
            for (x, y, conf) in points:
                if conf >= threshold:
                    key = (x, y)
                    filtered_tmp[label][key] = conf
                    semantic_lines.append(f"{label},({x},{y})\n")
            
                # Initialize Union-Find with all labels
        uf = UnionFind(list(filtered_tmp.keys()))

        # Compare all pairs of regions
        labels_list = list(filtered_tmp.keys())
        merge_count = 0

        OVERLAP_THRESHOLD = 0.5  # Merge if 50%+ overlap

        for i in range(len(labels_list)):
            for j in range(i + 1, len(labels_list)):
                label1 = labels_list[i]
                label2 = labels_list[j]
                
                region1 = set(filtered_tmp[label1])
                region2 = set(filtered_tmp[label2])
                
                #  Calculate overlap ratio
                overlap = self.compute_overlap_ratio(region1, region2)
                
                if overlap >= OVERLAP_THRESHOLD:
                    uf.union(label1, label2)
                    merge_count += 1
                    self.get_logger().info(
                        f"Merging '{label1}' and '{label2}' "
                        f"(overlap: {overlap:.2%}, cells: {len(region1)} & {len(region2)})"
                    )

        # Build final merged groups with accumulated confidence
        # TODO when merging, assign a lower confidence to the previously assigned grids. Prioritize the current detection.
        # start eliminating cells starting from the lowest confidence (tracking/decay/sensor drift)
        # also, the cells that are outside the boundinc box should be continuously assigned negative confidence.
        # oh, but this has an issue. what if the object is occluded in the current frame but visible in the previous frame?
        # or, the object is just not visible because of the light
        #basically should add confidence to all clusters and assumme the correct clusters have a significant higher confidence
        #use margin instead of fixed threshold??

        merged_groups = {}
        for label, cells_dict in filtered_tmp.items():
            root_label = uf.find(label)
            
            if root_label not in merged_groups:
                merged_groups[root_label] = {}
            
            # Accumulate confidence values for each cell
            for cell, conf in cells_dict.items():
                merged_groups[root_label][cell] = merged_groups[root_label].get(cell, 0.0) + conf


        self._recompute_geometry(merged_groups)



        # 5. Cube Marker & Text Marker pub
        markers = MarkerArray()
        marker_id = 0
        for label, cells in merged_groups.items():

            r, g, b = self.get_color_for_label(label)

            # セル群の重心を map 座標で計算
            sum_x = 0.0
            sum_y = 0.0
            count = 0

            for (i, j) in cells:
                cell_pos = self.cell_to_map(i, j)
                if cell_pos is None:
                    continue
                x_cell, y_cell = cell_pos
                sum_x += x_cell
                sum_y += y_cell
                count += 1

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
                cube.scale.x = self.map_info['resolution'] * 0.9
                cube.scale.y = self.map_info['resolution'] * 0.9
                cube.scale.z = 0.1
                cube.color.r = r
                cube.color.g = g
                cube.color.b = b
                cube.color.a = 1.0
                cube.lifetime.sec = 1
                cube.lifetime.nanosec = 0
                markers.markers.append(cube)

            if count == 0:
                continue

            centroid_x = sum_x / count
            centroid_y = sum_y / count

            display_text = self.format_display_label(label)

            text = Marker()
            text.header.frame_id = "map"
            text.header.stamp = self.get_clock().now().to_msg()
            text.ns = "semantic_text"
            text.id = marker_id
            marker_id += 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = centroid_x
            text.pose.position.y = centroid_y + self.map_info['resolution'] * 4.0
            text.pose.position.z = 1.0
            text.pose.orientation.w = 1.0
            text.scale.z = self.map_info['resolution'] * 4.0
            text.color.r = r
            text.color.g = g
            text.color.b = b
            text.color.a = 1.0
            text.text = display_text
            text.lifetime.sec = 1
            text.lifetime.nanosec = 0
            markers.markers.append(text)

        self.marker_pub.publish(markers)


    # カメラコールバック関数 (Modified to draw bounding boxes)
    def camera_callback(self, msg):

        try:
            # 画像の圧縮解凍
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return       
        
        # YOLO 検出
        results = self.yolo_model.track(cv_image, persist=True, classes=[0, 56, 60])
        bboxes = []
        
        # Create a copy for visualization
        annotated_image = cv_image.copy()
        
        if len(results) > 0:   
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf >= self.conf_threshold:
                    track_id = int(box.id[0]) if box.id is not None else -1
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    x2 = int(x2 - (x2-x1) * 0.0)
                    x1 = int(x1 + (x2 - x1) * 0.0)
                    cls_id = int(box.cls[0])
                    label = [name for name, cid in self.target_classes.items() if cid == cls_id][0]

                    # version include
                    # if not hasattr(self, 'untracked_counter'):
                    #     self.untracked_counter = 10000  # Start high to avoid conflicts
                    # track_id = int(box.id[0]) if box.id is not None else self.untracked_counter
                    # if box.id is None:
                    #     self.untracked_counter += 1
                    
                    # version skip
                    track_id = int(box.id[0]) if box.id is not None else -1
                    if track_id < 0:
                        continue  # Skip this detection
                    
                    combined_label = f"id:{track_id} {label}"  # Always has ID now
                    bboxes.append((x1, y1, x2, y2, combined_label, conf))
                    
                    # Draw bounding box on the image
                    color = (0, 255, 0) if label == 'person' else (255, 0, 0)
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    

                    # Draw label with confidence
                    label_text = f"{combined_label} {conf:.2f}"
                    
                    # Calculate text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    
                    # Draw background rectangle for text
                    cv2.rectangle(
                        annotated_image,
                        (x1, y1 - text_height - baseline - 5),
                        (x1 + text_width, y1),
                        color,
                        -1
                    )
                    
                    # Draw text
                    cv2.putText(
                        annotated_image,
                        label_text,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )
        
        if bboxes:

            # 🔹 Pass annotated_image to process LiDAR points
            self.process_one_frame_with_boxes(msg, bboxes, annotated_image)
        else:
            # 🔹 If no bboxes, still publish the image
            uncompressed_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            self.image_pub.publish(uncompressed_msg)
        
        ts = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        self.camera_msgs.append((msg, ts))

    # LiDAR コールバック関数
    def lidar_callback(self, msg):

        ts = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        point_cloud = np.stack([x, y], axis=1)
        self.lidar_msgs.append((msg, ts, point_cloud))

    # def lidar_callback(self, msg): #PointCloud2 version
    #     self.get_logger().info("LiDAR message received")
    #     ts = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
    #     points = list(point_cloud2.read_points(msg, field_names=("x", "y"), skip_nans=True) )
    #     if not points:
    #         return
    #     point_cloud = np.array(points)
    #     self.lidar_msgs.append((msg, ts, point_cloud))
    
    
    # Odometry コールバック関数
    def odometry_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.current_odometry = (x, y)
        # self.get_logger().info(f"Odometry updated: x={x:.2f}, y={y:.2f}")
    
    
    # バウンディングボックス内のLiDAR点群のマッチング処理
    def process_one_frame_with_boxes(self, cam_msg, bounding_boxes, annotated_image):
        try:
            if not self.lidar_msgs:
                self.get_logger().warn("No LiDAR message")
                # 🔹 Still publish the image even if no LiDAR
                uncompressed_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                self.image_pub.publish(uncompressed_msg)
                return
            lidar_msg, ts, point_cloud = self.lidar_msgs[-1]
            self.lidar_process_and_save(lidar_msg, bounding_boxes,
                                        cam_msg.header.stamp.sec, cam_msg.header.stamp.nanosec, 
                                        annotated_image)
        except Exception as e:
            self.get_logger().error(f"Error in process_one_frame_with_boxes: {e}")
    
    # LiDAR データ処理&保存
    def lidar_process_and_save(self, lidar_msg, bounding_boxes, sec, nsec, image=None):
        ranges = np.array(lidar_msg.ranges)
        angles = np.linspace(lidar_msg.angle_min, lidar_msg.angle_max, len(ranges))        
        lidar_4d = np.vstack([
            ranges * np.cos(angles),
            ranges * np.sin(angles),
            np.zeros_like(ranges),
            np.ones_like(ranges)
        ])
        
        # extrinsic変換
        cam_4d = self.extrinsic_matrix @ lidar_4d
        z = cam_4d[2, :]
        front = (z > 0)
        if not np.any(front):
            # Publish image even if no points
            if image is not None:
                uncompressed_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
                self.image_pub.publish(uncompressed_msg)
            return        
        cam_4d = cam_4d[:, front]
        
        # intrinsic変換
        uv = self.camera_matrix @ cam_4d[:3, :]
        uv /= cam_4d[2, :]
        pts_2d = uv[:2, :].T        
        
        # 🔹 Draw LiDAR points: GREEN if inside bbox, RED if outside
        if image is not None:
            for (px, py) in pts_2d:
                ix = int(px)
                iy = int(py)
                if 0 <= ix < self.width and 0 <= iy < self.height:
                    # Check if point is inside any bounding box
                    inside_bbox = False
                    for (x1, y1, x2, y2, combined_label, cconf) in bounding_boxes:
                        if x1 <= ix <= x2 and y1 <= iy <= y2:
                            inside_bbox = True
                            break
                    
                    # GREEN if inside bbox, RED if outside
                    color = (0, 255, 0) if inside_bbox else (0, 0, 255)
                    cv2.circle(image, (ix, iy), radius=2, color=color, thickness=-1)

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
        
        # Publish the image with both bounding boxes AND LiDAR points
        if image is not None:
            uncompressed_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            self.image_pub.publish(uncompressed_msg)    
        depths = [r for (r, th, _, _) in matched_pts_polar]
        min_depth = np.percentile(depths, 10)
        filtered = []
        for pt in matched_pts_polar:
            r, th, clabel, cconf = pt
            if r < min_depth + 0.3:
                filtered.append(pt)
        if matched_pts_polar:
            # (B) オドメトリ変化確認（移動しなければ保存しない）
            if self.current_odometry is not None:
                if self.last_saved_odometry is not None:
                    dx = self.current_odometry[0] - self.last_saved_odometry[0]
                    dy = self.current_odometry[1] - self.last_saved_odometry[1]
                    distance = math.sqrt(dx**2 + dy**2)
                    if distance < self.odom_threshold:
                        # self.get_logger().info("No odometry change")
                        return
                self.last_saved_odometry = self.current_odometry
            
            # (C) 点群をmap座標に変換後、lidar_detections.txtファイルに保存
            timestamp = f"{sec},{nsec}"
            detections_file = self.folder + "/lidar_detections.txt"
            try:
                with open(detections_file, "a") as f:
                    for pt in matched_pts_polar:
                        r, th, clabel, cconf = pt
                        # 信頼度フィルタリング
                        if cconf < self.conf_threshold:
                            continue                       
                        x_robot = r * math.sin(th + math.pi) + 0.156
                        y_robot = -r * math.cos(th + math.pi)
                        # x_robot = new_range * math.cos(new_angle)
                        # y_robot = new_range * math.sin(new_angle)   
                        
                        
                        # x_robot = r * math.cos(th)
                        # y_robot = r * math.sin(th)
                                             
                        try:
                            transform = self.tf_buffer.lookup_transform("map", "base_footprint", rclpy.time.Time())
                        except Exception as e:
                            self.get_logger().warn(f"TF lookup failure: {e}")
                            continue                       
                        x_map, y_map = self.transform_point(x_robot, y_robot, transform)
                        line = f"{timestamp},{clabel},{cconf:.2f},{x_map:.3f},{y_map:.3f}\n"
                        f.write(line)
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
        # yaw=2*math.asin(q.z)
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
            # self.get_logger().info(f"Tracking box created: {combined_label}")
        # self.get_logger().info("Tracking result update complete")

    def _recompute_geometry(self, merged_groups):
        """Extract all cell positions and compute convex hull"""
        
        # Collect all cells from all labels
        all_points = []
        for label, cells_dict, confidence in merged_groups.items():
            for (i, j) in cells_dict.keys():
                # Convert grid cell to map coordinates
                map_pos = self.cell_to_map(i, j)
                if map_pos is not None:
                    all_points.append(map_pos)
        
        if len(all_points) < 3:
            self.get_logger().warn(f"Not enough points for hull: {len(all_points)}")
            return
        
        points = np.array(all_points)
        
        # Compute Convex Hull
        try:
            if np.unique(points, axis=0).shape[0] >= 3:
                hull = ConvexHull(points)
                self.hull_vertices = points[hull.vertices]
                
                # self.get_logger().info(f"Hull computed with {len(self.hull_vertices)} vertices")
                self._publish_hull_visualization()
        except Exception as e:
            self.get_logger().error(f"Hull computation failed: {e}")
            self.hull_vertices = points

    def classify_points(self, candidate_points):
        """
        Classifies new points as:
        0: Inside Hull (Part of Object)
        1: Buffer Zone (Near boundary, ignore)
        2: Unknown Seed (Outside boundary, use for new clusters)
        """
        if self.hull_vertices is None or len(self.hull_vertices) < 3:
             # Fallback: Everything is unknown if we haven't learned the object yet
            return np.full(len(candidate_points), 2)

        from matplotlib.path import Path
        hull_path = Path(self.hull_vertices)

        is_inside = hull_path.contains_points(candidate_points)
        labels = np.zeros(len(candidate_points), dtype=int)

        for i, p in enumerate(candidate_points):
            if is_inside[i]:
                labels[i] = 0 # Object
            else:
                # Calculate min distance to hull vertices for buffer check
                dists = np.linalg.norm(self.hull_vertices - p, axis=1)
                min_dist = np.min(dists)

                if min_dist < self.expansion_buffer:
                    labels[i] = 1 # Buffer/Gap
                else:
                    labels[i] = 2 # Seed for Unknown
        return labels   
    
    def _recompute_geometry(self, merged_groups):
        """Compute a convex hull for each cluster/label"""
        
        self.hull_vertices = {}  # Store multiple hulls: {label: vertices}
        
        for label, cells_dict in merged_groups.items():
            # Collect points for this specific label
            label_points = []
            for (i, j) in cells_dict.keys():
                map_pos = self.cell_to_map(i, j)
                if map_pos is not None:
                    label_points.append(map_pos)
            
            if len(label_points) < 3:
                self.get_logger().warn(f"Label '{label}': Not enough points for hull ({len(label_points)})")
                continue
            
            points = np.array(label_points)
            
            # Compute Convex Hull for this label
            try:
                if np.unique(points, axis=0).shape[0] >= 3:
                    hull = ConvexHull(points)
                    self.hull_vertices[label] = points[hull.vertices]
                    # self.get_logger().info(f"Label '{label}': Hull with {len(self.hull_vertices[label])} vertices")
            except Exception as e:
                self.get_logger().error(f"Label '{label}': Hull failed: {e}")
                self.hull_vertices[label] = points
        
        # Publish all hulls
        if self.hull_vertices:
            self._publish_hull_visualization()


    def _publish_hull_visualization(self):
        """Publish a separate hull marker for each label"""
        
        if not self.hull_vertices:
            self.get_logger().warn("No hull vertices to publish")
            return
        
        marker_array = MarkerArray()
        marker_id = 0
        
        for label, vertices in self.hull_vertices.items():
            if len(vertices) < 3:
                continue
            
            # Get color for this label (same as your semantic markers)
            r, g, b = self.get_color_for_label(label)
            
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "convex_hulls"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            
            # Line properties
            marker.scale.x = 0.08  # Line width
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 1.0
            marker.lifetime.sec = 1
            marker.lifetime.nanosec = 0
            
            # Add hull vertices as a closed loop
            for vertex in vertices:
                p = Point()
                p.x = float(vertex[0])
                p.y = float(vertex[1])
                p.z = 0.1  # Slightly above ground
                marker.points.append(p)
            
            # Close the loop
            p = Point()
            p.x = float(vertices[0][0])
            p.y = float(vertices[0][1])
            p.z = 0.1
            marker.points.append(p)
            
            marker_array.markers.append(marker)
        
        # Change publisher to MarkerArray
        self.hull_viz_pub.publish(marker_array)
    def compute_overlap_ratio(self, region1, region2):
        """
        Compute overlap ratio between two regions.
        Returns the ratio of intersection to the smaller region.
        """
        intersection = region1.intersection(region2)
        if len(intersection) == 0:
            return 0.0
        
        # Overlap coefficient: intersection / min(region sizes)
        # This means if small region is 80% contained in large region, ratio = 0.8
        min_size = min(len(region1), len(region2))
        overlap_ratio = len(intersection) / min_size
        
        return overlap_ratio
    
def main(args=None):
    rclpy.init(args=args)
    node = LidarCamBasePolarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
