import cv2
import numpy as np

def detect_refined_rear_view(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get image dimensions
    height, width = gray.shape
    
    # Focus on the middle portion of the image where rear view is likely to be
    roi_height = int(height * 0.4)  # Take 40% of image height
    roi_y_start = int(height * 0.3)  # Start from 30% down from top
    
    # Create ROI (Region of Interest)
    roi = gray[roi_y_start:roi_y_start + roi_height, :]
    
    # Enhance contrast in ROI
    roi = cv2.equalizeHist(roi)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply horizontal gradient detection
    sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    
    # Threshold gradient image
    _, grad_binary = cv2.threshold(scaled_sobel, 30, 255, cv2.THRESH_BINARY)
    
    # Combine binary images
    combined_binary = cv2.bitwise_and(binary, grad_binary)
    
    # Find contours in ROI
    contours, _ = cv2.findContours(
        combined_binary, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter and process contours
    valid_x_coords = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(contour)
            # Add the x-coordinates of valid contours
            valid_x_coords.extend([x, x + w])
    
    if valid_x_coords:
        # Calculate bounding box coordinates
        x_min = max(min(valid_x_coords), 0)
        x_max = min(max(valid_x_coords), width)
        
        # Ensure minimum width (adjust as needed)
        min_width = width * 0.2  # 20% of image width
        current_width = x_max - x_min
        if current_width < min_width:
            center = (x_min + x_max) // 2
            x_min = max(center - min_width//2, 0)
            x_max = min(center + min_width//2, width)
        
        # Draw the bounding box
        result = image.copy()
        cv2.rectangle(
            result, 
            (x_min, roi_y_start), 
            (x_max, roi_y_start + roi_height), 
            (0, 255, 0), 
            2
        )
        
        return result, (x_min, roi_y_start, x_max - x_min, roi_height)
    
    return image, None

def main():
    image_path = 'car_rear.png'
    result_image, bbox = detect_refined_rear_view(image_path)
    
    if bbox:
        print(f"Bounding box coordinates: {bbox}")
        cv2.imshow('Refined Rear View Detection', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No rear view detected")

if __name__ == "__main__":
    main()