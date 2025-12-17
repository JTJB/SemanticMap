#!/usr/bin/env python3
import cv2
import numpy as np
import yaml
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import sys

def run_dbscan_on_map(map_file, yaml_file):
    # Load the map
    img = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)
    
    # Load metadata
    with open(yaml_file, 'r') as f:
        map_metadata = yaml.safe_load(f)
    
    resolution = map_metadata['resolution']
    origin = map_metadata['origin']
    
    # Get occupied points (obstacles)
    occupied = np.where(img < 120)  # Black pixels = obstacles
    
    # Convert to real-world coordinates (meters)
    x_coords = occupied[1] * resolution + origin[0]
    y_coords = (img.shape[0] - occupied[0]) * resolution + origin[1]
    
    # Stack into points array
    points = np.column_stack((x_coords, y_coords))
    print(f"Total points: {len(points)}")
    
    # Run DBSCAN
    # eps = maximum distance between two points to be in same cluster
    # min_samples = minimum points to form a cluster
    dbscan = DBSCAN(eps=0.2, min_samples=70)
    labels = dbscan.fit_predict(points)
    
    # Get number of clusters (excluding noise = -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Clusters found: {n_clusters}")
    print(f"Noise points: {n_noise}")
    
    # Visualize results
    plt.figure(figsize=(20, 15))
    
    # Generate random colors for each cluster
    unique_labels = set(labels)
    np.random.seed(42)  # For reproducibility
    colors = np.random.rand(len(unique_labels), 3)
    
    for i, k in enumerate(unique_labels):
        if k == -1:
            # Noise points in black
            col = [0, 0, 0]
        else:
            col = colors[i]
        
        class_member_mask = (labels == k)
        xy = points[class_member_mask]

        # bounding box
        xmin = xy[:,0].min()
        xmax = xy[:,0].max()
        ymin = xy[:,1].min()
        ymax = xy[:,1].max()

        width  = xmax - xmin
        height = ymax - ymin

        # size filter
        if not (0 <= width <= 0.4 and 0 <= height <= 0.8):
            continue
        
        mu = xy.mean(axis=0)
        U, S, Vt = np.linalg.svd(xy - mu, full_matrices=False)
        ratio = S[1] / S[0]

        if ratio < 0.5:
            continue  # line-like cluster

        
        plt.gca().add_patch(
            plt.Rectangle(
                (xmin, ymin),
                width,
                height,
                fill=False,
                edgecolor='red',
                linewidth=1
            )
        )


        plt.scatter(xy[:, 0], xy[:, 1], 
                   c=[col], 
                   s=1,
                   edgecolors='none')
    
    plt.title(f'DBSCAN Clustering: {n_clusters} clusters found')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('dbscan_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return points, labels

if __name__ == '__main__':
    # Run DBSCAN
    map_file = sys.argv[1] + '.pgm' if len(sys.argv) > 1 else 'my_map.pgm'
    yaml_file = sys.argv[1] + '.yaml' if len(sys.argv) > 1 else 'my_map.yaml'
    print("Map file: ", map_file, ", YAML file:", yaml_file)
    points, labels = run_dbscan_on_map(map_file, yaml_file)