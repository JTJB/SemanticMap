#!/usr/bin/env python3
import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import RANSACRegressor
from scipy.spatial.distance import cdist

def fit_line_ransac(points, residual_threshold=0.1, min_samples=10):
    """
    Fit a line to points using RANSAC
    Returns: (inliers_mask, slope, intercept, score)
    """
    if len(points) < min_samples:
        return None, None, None, 0
    
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    
    try:
        ransac = RANSACRegressor(
            residual_threshold=residual_threshold,
            min_samples=min_samples,
            max_trials=1000,
            random_state=42
        )
        ransac.fit(X, y)
        
        inlier_mask = ransac.inlier_mask_
        slope = ransac.estimator_.coef_[0]
        intercept = ransac.estimator_.intercept_
        score = inlier_mask.sum() / len(points)
        
        return inlier_mask, slope, intercept, score
    except:
        return None, None, None, 0

def fit_vertical_line_ransac(points, residual_threshold=0.3, min_samples=60):
    """
    Fit a vertical line (x = c) using RANSAC
    Returns: (inliers_mask, x_value, score)
    """
    if len(points) < min_samples:
        return None, None, 0
    
    # For vertical lines, flip X and y
    X = points[:, 1].reshape(-1, 1)
    y = points[:, 0]
    
    try:
        ransac = RANSACRegressor(
            residual_threshold=residual_threshold,
            min_samples=min_samples,
            max_trials=1000,
            random_state=42
        )
        ransac.fit(X, y)
        
        inlier_mask = ransac.inlier_mask_
        x_value = ransac.estimator_.intercept_
        score = inlier_mask.sum() / len(points)
        
        return inlier_mask, x_value, score
    except:
        return None, None, 0

def detect_line_segment(points, residual_threshold=0.1):
    """
    Detect if points form a line (either regular or vertical)
    Returns: (is_vertical, inliers_mask, params, score)
    """
    # Try regular line fit (y = mx + b)
    inliers_reg, slope, intercept, score_reg = fit_line_ransac(
        points, residual_threshold=residual_threshold
    )
    
    # Try vertical line fit (x = c)
    inliers_vert, x_value, score_vert = fit_vertical_line_ransac(
        points, residual_threshold=residual_threshold
    )
    
    # Choose the better fit
    if score_reg > score_vert:
        return False, inliers_reg, (slope, intercept), score_reg
    else:
        return True, inliers_vert, x_value, score_vert

def extract_line_clusters(points, residual_threshold=0.1, min_inliers=20, max_iterations=50):
    """
    Iteratively extract line segments from points using RANSAC
    Returns: list of (cluster_points, line_params, is_vertical)
    """
    remaining_points = points.copy()
    remaining_indices = np.arange(len(points))
    clusters = []
    
    for iteration in range(max_iterations):
        if len(remaining_points) < min_inliers:
            break
        
        # Detect line in remaining points
        is_vertical, inliers_mask, params, score = detect_line_segment(
            remaining_points, residual_threshold=residual_threshold
        )
        
        if inliers_mask is None or inliers_mask.sum() < min_inliers:
            break
        
        # Extract inlier points
        cluster_indices = remaining_indices[inliers_mask]
        cluster_points = points[cluster_indices]
        
        clusters.append({
            'points': cluster_points,
            'indices': cluster_indices,
            'is_vertical': is_vertical,
            'params': params,
            'score': score,
            'size': len(cluster_points)
        })
        
        print(f"Iteration {iteration + 1}: Found line with {len(cluster_points)} points "
              f"({'vertical' if is_vertical else 'angled'}, score: {score:.3f})")
        
        # Remove inliers from remaining points
        remaining_points = remaining_points[~inliers_mask]
        remaining_indices = remaining_indices[~inliers_mask]
    
    # Handle remaining points as noise
    if len(remaining_points) > 0:
        clusters.append({
            'points': remaining_points,
            'indices': remaining_indices,
            'is_vertical': None,
            'params': None,
            'score': 0,
            'size': len(remaining_points)
        })
        print(f"Remaining {len(remaining_points)} points marked as noise")
    
    return clusters

def merge_parallel_lines(clusters, angle_threshold=np.pi/12, distance_threshold=0.5):
    """
    Merge clusters that represent parallel line segments close to each other
    """
    if len(clusters) <= 1:
        return clusters
    
    merged = [False] * len(clusters)
    merged_clusters = []
    
    for i in range(len(clusters)):
        if merged[i] or clusters[i]['is_vertical'] is None:  # Skip noise
            continue
        
        current_group = [i]
        
        for j in range(i + 1, len(clusters)):
            if merged[j] or clusters[j]['is_vertical'] is None:
                continue
            
            # Check if both are vertical or both are angled
            if clusters[i]['is_vertical'] != clusters[j]['is_vertical']:
                continue
            
            # For vertical lines
            if clusters[i]['is_vertical']:
                x1 = clusters[i]['params']
                x2 = clusters[j]['params']
                
                # Check if they're close
                if abs(x1 - x2) < distance_threshold:
                    current_group.append(j)
                    merged[j] = True
            
            # For angled lines
            else:
                slope1, intercept1 = clusters[i]['params']
                slope2, intercept2 = clusters[j]['params']
                
                # Calculate angles
                angle1 = np.arctan(slope1)
                angle2 = np.arctan(slope2)
                angle_diff = abs(angle1 - angle2)
                
                # Check if parallel
                if angle_diff < angle_threshold:
                    # Check perpendicular distance between lines
                    # Distance from point to line: |ax + by + c| / sqrt(a^2 + b^2)
                    # Line: y = mx + b => mx - y + b = 0
                    
                    # Use centroid of cluster j to measure distance to line i
                    centroid = clusters[j]['points'].mean(axis=0)
                    x0, y0 = centroid
                    
                    a, b, c = slope1, -1, intercept1
                    dist = abs(a * x0 + b * y0 + c) / np.sqrt(a**2 + b**2)
                    
                    if dist < distance_threshold:
                        current_group.append(j)
                        merged[j] = True
        
        # Merge the group
        if len(current_group) > 1:
            print(f"Merging {len(current_group)} parallel line segments")
        
        merged[i] = True
        all_points = np.vstack([clusters[idx]['points'] for idx in current_group])
        all_indices = np.concatenate([clusters[idx]['indices'] for idx in current_group])
        
        merged_clusters.append({
            'points': all_points,
            'indices': all_indices,
            'is_vertical': clusters[i]['is_vertical'],
            'params': clusters[i]['params'],
            'score': clusters[i]['score'],
            'size': len(all_points)
        })
    
    # Add noise cluster if exists
    noise_clusters = [c for i, c in enumerate(clusters) if c['is_vertical'] is None]
    merged_clusters.extend(noise_clusters)
    
    return merged_clusters

def run_ransac_on_map(map_file, yaml_file, residual_threshold=0.1, min_inliers=20):
    # Load the map
    img = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)
    
    # Load metadata
    with open(yaml_file, 'r') as f:
        map_metadata = yaml.safe_load(f)
    
    resolution = map_metadata['resolution']
    origin = map_metadata['origin']
    
    # Get occupied points (obstacles)
    occupied = np.where(img < 110)  # Black pixels = obstacles
    
    # Convert to real-world coordinates (meters)
    x_coords = occupied[1] * resolution + origin[0]
    y_coords = (img.shape[0] - occupied[0]) * resolution + origin[1]
    
    # Stack into points array
    points = np.column_stack((x_coords, y_coords))
    print(f"Total points: {len(points)}")
    
    # Extract line clusters using RANSAC
    print("\nExtracting line segments with RANSAC...")
    clusters = extract_line_clusters(
        points, 
        residual_threshold=residual_threshold,
        min_inliers=min_inliers,
        max_iterations=50
    )
    
    print(f"\nInitial line segments found: {len(clusters)}")
    
    # Merge parallel lines
    print("\nMerging parallel line segments...")
    clusters = merge_parallel_lines(clusters)
    
    n_clusters = len([c for c in clusters if c['is_vertical'] is not None])
    n_noise = sum([c['size'] for c in clusters if c['is_vertical'] is None])
    
    print(f"\nFinal clusters: {n_clusters}")
    print(f"Noise points: {n_noise}")
    
    # Visualize results
    plt.figure(figsize=(20, 15))
    
    # Generate random colors for each cluster
    np.random.seed(42)
    colors = np.random.rand(len(clusters), 3)
    
    for i, cluster in enumerate(clusters):
        if cluster['is_vertical'] is None:
            # Noise in black
            col = [0, 0, 0]
        else:
            col = colors[i]
        
        plt.scatter(cluster['points'][:, 0], cluster['points'][:, 1],
                   c=[col],
                   s=1,
                   edgecolors='none',
                   label=f"Line {i+1} ({cluster['size']} pts)" if cluster['is_vertical'] is not None else "Noise")
    
    plt.title(f'RANSAC Line Detection: {n_clusters} line segments found')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ransac_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return points, clusters

if __name__ == '__main__':
    # Run RANSAC
    map_file = sys.argv[1] + '.pgm' if len(sys.argv) > 1 else 'my_map.pgm'
    yaml_file = sys.argv[1] + '.yaml' if len(sys.argv) > 1 else 'my_map.yaml'
    print("Map file: ", map_file, ", YAML file:", yaml_file)
    
    # Adjust these parameters based on your map
    # residual_threshold: how far a point can be from the line (in meters)
    # min_inliers: minimum points to form a line segment
    points, clusters = run_ransac_on_map(
        map_file, 
        yaml_file,
        residual_threshold=0.1,  # 10cm tolerance
        min_inliers=20           # at least 20 points per line
    )