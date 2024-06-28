import cv2
import numpy as np

# Load the image
img_path = './scan/datas/BEST_V2/'
img_name = '0003.jpg'
image = cv2.imread(f"{img_path}{img_name}", cv2.IMREAD_COLOR)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve contour detection
blurred = cv2.GaussianBlur(gray, (1, 1), 0)

# Detect edges using Canny
edges = cv2.Canny(blurred, 1, 1)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
for contour in contours:
    # Approximate contour to a circle
    if len(contour) >= 5:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if 10 < radius < 30:  # Assuming circles with numbers have a radius in this range
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(image, center, radius, (0, 0, 255), 2)

# Save the image with contours
output_path_cv = f"{img_path}image_with_contours.png"
cv2.imwrite(output_path_cv, gray)

output_path_cv
