import cv2
import numpy as np
import matplotlib.pyplot as plt


class TaillightDetector:
    def __init__(self,
                 status_change_threshold=15,
                 red_threshold_lower1=(0, 100, 100),
                 red_threshold_upper1=(10, 255, 255),
                 red_threshold_lower2=(160, 100, 100),
                 red_threshold_upper2=(180, 255, 255),
                 min_contour_area=50,
                 morphological_kernel_size=5,
                 dilation_iterations=2,
                 symmetric_distance_threshold=20):
        """
        Initialize the taillight detector with configurable parameters

        Args:
            status_change_threshold: Threshold for determining brake state change
            red_threshold_lower1: Lower bound for red color in HSV (first range)
            red_threshold_upper1: Upper bound for red color in HSV (first range)
            red_threshold_lower2: Lower bound for red color in HSV (second range)
            red_threshold_upper2: Upper bound for red color in HSV (second range)
            min_contour_area: Minimum area for considering a contour valid
            morphological_kernel_size: Size of kernel for morphological operations
            dilation_iterations: Number of dilation iterations
            symmetric_distance_threshold: Maximum y-distance for considering two lights as a pair
        """
        self.status_change_threshold = status_change_threshold
        self.red_threshold_lower1 = np.array(red_threshold_lower1)
        self.red_threshold_upper1 = np.array(red_threshold_upper1)
        self.red_threshold_lower2 = np.array(red_threshold_lower2)
        self.red_threshold_upper2 = np.array(red_threshold_upper2)
        self.min_contour_area = min_contour_area
        self.morphological_kernel_size = morphological_kernel_size
        self.dilation_iterations = dilation_iterations
        self.symmetric_distance_threshold = symmetric_distance_threshold

        # Store previous brightness values for comparison (for video sequences)
        self.previous_brightness = None

    def detect_taillights(self, image, visualize=False):
        """
        Detect taillights in a vehicle rear image

        Args:
            image: Input BGR image
            visualize: Whether to display visualization of intermediate steps

        Returns:
            bounding_boxes: List of bounding boxes for detected taillights [(x1,y1,x2,y2),...]
            brake_status: Boolean indicating if brake lights are on
            brightness: Current brightness value of detected taillights
        """
        # Make a copy of the input image
        car_img_rgb = image.copy()

        # 1. Convert to YCrCb color space (better for detecting red components)
        img_yCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # 2. Apply threshold to extract potential taillight regions
        threshold_img = self._get_threshold_img(img_yCrCb)

        # 3. Apply morphological operations to clean up the mask
        morpho_img = self._morphological_operations(threshold_img)

        # 4. Find connected components
        connectivity = 4
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            morpho_img, connectivity, cv2.CV_32S
        )

        # 5. Identify potential light pairs based on symmetry
        light_pairs = self._find_symmetrical_pairs(
            morpho_img, n_labels, labels, stats, centroids
        )

        # 6. Find the best pair (largest surface area)
        best_pair, mean_brightness = self._find_best_pair(
            light_pairs, labels, img_yCrCb
        )

        # 7. Get bounding boxes for the detected taillights
        bounding_boxes = self._get_bounding_boxes(best_pair, labels)

        # 8. Determine brake status by comparing with previous brightness
        brake_status = self._determine_brake_status(mean_brightness)

        # 9. Visualize results if requested
        if visualize:
            self._visualize_detection(
                image, img_yCrCb, threshold_img, morpho_img,
                labels, best_pair, bounding_boxes, brake_status
            )

        return bounding_boxes, brake_status, mean_brightness

    def _get_threshold_img(self, img_yCrCb):
        """Extract potential taillight regions using YCrCb thresholding"""
        # Extract Y and Cr channels
        car_img_Y = img_yCrCb[:, :, 0]
        car_img_Cr = img_yCrCb[:, :, 1]

        # Apply adaptive thresholding to the Cr channel (red chrominance)
        block_size = 11
        c_value = 7
        th_Cr = cv2.adaptiveThreshold(
            car_img_Cr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, c_value
        )

        return th_Cr

    def _morphological_operations(self, img):
        """Apply morphological operations to clean up the mask"""
        # Erosion to remove small noise
        kernel_small = np.ones((2, 2), np.uint8)
        erosion = cv2.erode(img, kernel_small, iterations=1)

        # Dilation to connect nearby regions
        kernel_large = np.ones((self.morphological_kernel_size, self.morphological_kernel_size), np.uint8)
        dilation = cv2.dilate(erosion, kernel_large, iterations=self.dilation_iterations)

        return dilation

    def _find_symmetrical_pairs(self, img, n_labels, labels, stats, centroids):
        """Find potential taillight pairs based on symmetry"""
        light_pairs = []

        # Check all possible pairs of connected components
        for i in range(1, n_labels):
            for j in range(i + 1, n_labels):
                i_cent_x, i_cent_y = int(centroids[i, 0]), int(centroids[i, 1])
                j_cent_x, j_cent_y = int(centroids[j, 0]), int(centroids[j, 1])

                # Check if the y-coordinates are similar (horizontally aligned)
                if abs(i_cent_y - j_cent_y) < self.symmetric_distance_threshold:
                    # This is a potential taillight pair
                    light_pairs.append([i, j])

        return light_pairs

    def _find_best_pair(self, light_pairs, labels, img_yCrCb):
        """Find the best taillight pair (largest surface area)"""
        if not light_pairs:
            return [], 0

        best_pair = []
        max_surface = 0
        max_brightness = 0

        for pair in light_pairs:
            part_1, part_2 = pair

            # Count pixels belonging to each component
            count1 = np.sum(labels == part_1)
            count2 = np.sum(labels == part_2)
            surface_sum = count1 + count2

            if surface_sum > max_surface:
                best_pair = pair
                max_surface = surface_sum

                # Calculate mean brightness (Y channel) and redness (Cr channel)
                cr_channel = img_yCrCb[:, :, 1].copy()
                y_channel = img_yCrCb[:, :, 0].copy()

                # Calculate mean Y value (brightness) of the pair
                brightness1 = np.mean(y_channel[labels == part_1])
                brightness2 = np.mean(y_channel[labels == part_2])
                mean_brightness = np.mean([brightness1, brightness2])
                max_brightness = mean_brightness

        return best_pair, max_brightness

    def _get_bounding_boxes(self, pair, labels):
        """Get bounding boxes for the detected taillights"""
        if not pair:
            return []

        bounding_boxes = []

        for part in pair:
            # Find coordinates of all pixels belonging to this component
            y_coords, x_coords = np.where(labels == part)

            if len(y_coords) > 0 and len(x_coords) > 0:
                # Calculate bounding box
                xmin, ymin = np.min(x_coords), np.min(y_coords)
                xmax, ymax = np.max(x_coords), np.max(y_coords)

                # Add to list of bounding boxes
                bounding_boxes.append([xmin, ymin, xmax, ymax])

        return bounding_boxes

    def _determine_brake_status(self, current_brightness):
        """Determine if brake lights are on based on brightness comparison"""
        if self.previous_brightness is None:
            # First frame, can't determine status yet
            self.previous_brightness = current_brightness
            return False

        # If current brightness is significantly higher than previous, brake is likely on
        brake_on = (current_brightness - self.previous_brightness) > self.status_change_threshold

        # Update previous brightness for next comparison
        self.previous_brightness = current_brightness

        return brake_on

    def _visualize_detection(self, original, ycrcb, threshold, morpho, labels,
                             best_pair, bounding_boxes, brake_status):
        """Visualize intermediate steps and final detection"""
        # Create visualization of YCrCb channels
        y_channel = ycrcb[:, :, 0]
        cr_channel = ycrcb[:, :, 1]
        cb_channel = ycrcb[:, :, 2]

        # Create labeled image for visualization
        height, width = original.shape[:2]
        labeled_img = np.zeros((height, width, 3), dtype=np.uint8)

        # Visualize best pair
        if best_pair:
            for part in best_pair:
                # Color the regions of the best pair
                labeled_img[labels == part] = [0, 0, 255]  # Red color

        # Copy original image for drawing bounding boxes
        result_img = original.copy()

        # Draw bounding boxes
        for box in bounding_boxes:
            xmin, ymin, xmax, ymax = box
            color = (0, 0, 255) if brake_status else (0, 255, 0)  # Red if braking, green otherwise
            cv2.rectangle(result_img, (xmin, ymin), (xmax, ymax), color, 2)

        # Add text indicating brake status
        status_text = "BRAKE ON" if brake_status else "BRAKE OFF"
        cv2.putText(result_img, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if brake_status else (0, 255, 0), 2)

        # Display images
        plt.figure(figsize=(15, 10))

        plt.subplot(3, 3, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(3, 3, 2)
        plt.imshow(y_channel, cmap='gray')
        plt.title('Y Channel')

        plt.subplot(3, 3, 3)
        plt.imshow(cr_channel, cmap='gray')
        plt.title('Cr Channel (Red)')

        plt.subplot(3, 3, 4)
        plt.imshow(threshold, cmap='gray')
        plt.title('Threshold Image')

        plt.subplot(3, 3, 5)
        plt.imshow(morpho, cmap='gray')
        plt.title('Morphological Operations')

        plt.subplot(3, 3, 6)
        plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
        plt.title('Detected Light Regions')

        plt.subplot(3, 3, 7)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title('Final Detection')

        plt.tight_layout()
        plt.show()


def detect_brake_lights(image_path, visualize=False):
    """
    Detect brake lights in a single image

    Args:
        image_path: Path to the input image
        visualize: Whether to display visualization

    Returns:
        Image with annotated detections
    """
    # Create detector
    detector = TaillightDetector()

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Detect taillights
    bounding_boxes, brake_status, brightness = detector.detect_taillights(image, visualize)

    # Annotate image
    result = image.copy()
    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = box
        color = (0, 0, 255) if brake_status else (0, 255, 0)  # Red if braking, green otherwise
        cv2.rectangle(result, (xmin, ymin), (xmax, ymax), color, 2)

    # Add text indicating brake status
    status_text = "BRAKE ON" if brake_status else "BRAKE OFF"
    cv2.putText(result, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if brake_status else (0, 255, 0), 2)

    return result, brake_status, bounding_boxes


def process_video(video_path, output_path, visualize=False):
    """
    Process a video to detect brake lights in each frame

    Args:
        video_path: Path to the input video
        output_path: Path to save the output video
        visualize: Whether to display visualization of each frame
    """
    # Create detector
    detector = TaillightDetector()

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    brake_status_history = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect taillights
            bounding_boxes, brake_status, brightness = detector.detect_taillights(frame, False)
            brake_status_history.append(brake_status)

            # Annotate frame
            result = frame.copy()
            for box in bounding_boxes:
                xmin, ymin, xmax, ymax = box
                color = (0, 0, 255) if brake_status else (0, 255, 0)  # Red if braking, green otherwise
                cv2.rectangle(result, (xmin, ymin), (xmax, ymax), color, 2)

            # Add text indicating brake status
            status_text = "BRAKE ON" if brake_status else "BRAKE OFF"
            cv2.putText(result, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if brake_status else (0, 255, 0), 2)

            # Write frame to output video
            out.write(result)

            # Display frame if requested
            if visualize:
                cv2.imshow('Frame', result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1

            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")

    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    return brake_status_history


if __name__ == "__main__":
    # Example usage for a single image
    image_path = "dataset/sample_images/test_image_2.png"
    result, brake_status, boxes = detect_brake_lights(image_path, visualize=True)

    # Display and save result
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.imwrite("result.jpg", result)

    # Example usage for a video
    # video_path = "dataset/sample_video.mp4"
    # output_path = "result_video.mp4"
    # brake_status_history = process_video(video_path, output_path, visualize=True)