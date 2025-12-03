import cv2
import numpy as np
import sys

# Replace with the actual path to your PGM file
image_path = sys.argv[1] if len(sys.argv) > 1 else 'my_map.pgm'

# Read the PGM image as is
img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

if img is None:
    print(f"Error: Could not open or find the image at {image_path}")
else:
    # Check if normalization is needed (e.g., if it's a 16-bit PGM)
    if img.dtype != 'uint8':
        # Find the maximum pixel value in the image for normalization
        # Alternatively, you could parse the PGM header for the official max_val
        max_val = img.max() 
        
        # Normalize the image to 8-bit for display
        img_display = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    else:
        img_display = img

    # Display the image
    cv2.imshow("PGM Image", img_display)

    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()