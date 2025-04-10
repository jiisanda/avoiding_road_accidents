import cv2
import os
import numpy as np

def detect_taillights(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define red color range for taillights
    # We need two ranges because red wraps around in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red regions
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area
    min_area = 50  # Reduced minimum area to catch smaller lights
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if valid_contours:
        # Find the extreme points of all contours
        all_x = []
        all_y = []
        for contour in valid_contours:
            for point in contour:
                all_x.append(point[0][0])
                all_y.append(point[0][1])
        
        # Calculate the bounding box coordinates
        x_min = max(min(all_x) - 10, 0)  # Add padding but don't go below 0
        x_max = min(max(all_x) + 10, image.shape[1])  # Add padding but don't exceed image width
        y_min = max(min(all_y) - 5, 0)  # Add padding but don't go below 0
        y_max = min(max(all_y) + 5, image.shape[0])  # Add padding but don't exceed image height
        
        # Draw the bounding box
        result = image.copy()
        cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        return result, (x_min, y_min, x_max - x_min, y_max - y_min)
    
    return image, None

def main():
    test_dir = 'data/test'
    supported_formats = ('.png', '.jpg', '.jpeg')

    for filename in os.listdir(test_dir):
        if filename.lower().endswith(supported_formats):
            image_path = os.path.join(test_dir, filename)
            print(f"\nProcessing {filename}...")

            result_image, bbox = detect_taillights(image_path)

            if bbox:
                print(f"Bounding box coordinates for {filename}: {bbox}")
                cv2.imshow(f'Detected Taillights - {filename}', result_image)
                cv2.waitKey(0)
            else:
                print(f"No taillights detected in {filename}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
