import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tail_light_detector import TaillightDetector

# Parameters to test
PARAM_CONFIGS = [
    # Default parameters
    {
        "name": "Default",
        "status_change_threshold": 15,
        "red_threshold_lower1": (0, 100, 100),
        "red_threshold_upper1": (10, 255, 255),
        "red_threshold_lower2": (160, 100, 100),
        "red_threshold_upper2": (180, 255, 255),
        "min_contour_area": 50,
        "morphological_kernel_size": 5,
        "dilation_iterations": 2,
        "symmetric_distance_threshold": 20
    },
    # Enhanced red detection
    {
        "name": "Enhanced Red",
        "status_change_threshold": 15,
        "red_threshold_lower1": (0, 80, 80),
        "red_threshold_upper1": (15, 255, 255),
        "red_threshold_lower2": (150, 80, 80),
        "red_threshold_upper2": (180, 255, 255),
        "min_contour_area": 40,
        "morphological_kernel_size": 5,
        "dilation_iterations": 2,
        "symmetric_distance_threshold": 20
    },
    # Increased sensitivity
    {
        "name": "High Sensitivity",
        "status_change_threshold": 10,
        "red_threshold_lower1": (0, 70, 70),
        "red_threshold_upper1": (15, 255, 255),
        "red_threshold_lower2": (150, 70, 70),
        "red_threshold_upper2": (180, 255, 255),
        "min_contour_area": 30,
        "morphological_kernel_size": 7,
        "dilation_iterations": 3,
        "symmetric_distance_threshold": 30
    },
    # Reduced noise
    {
        "name": "Low Noise",
        "status_change_threshold": 20,
        "red_threshold_lower1": (0, 120, 120),
        "red_threshold_upper1": (10, 255, 255),
        "red_threshold_lower2": (165, 120, 120),
        "red_threshold_upper2": (180, 255, 255),
        "min_contour_area": 80,
        "morphological_kernel_size": 3,
        "dilation_iterations": 1,
        "symmetric_distance_threshold": 15
    }
]


def test_detector_on_images(image_dir, output_dir=None, show_results=True):
    """
    Test the taillight detector with different parameter configurations on a set of images

    Args:
        image_dir: Directory containing test images
        output_dir: Directory to save output images (optional)
        show_results: Whether to display visualization
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No image files found in {image_dir}")
        return

    print(f"Found {len(image_files)} images")

    # Process each image with each parameter configuration
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        # Create figure to display results
        if show_results:
            plt.figure(figsize=(15, 5 * len(PARAM_CONFIGS)))

        for i, config in enumerate(PARAM_CONFIGS):
            # Create detector with current configuration
            detector = TaillightDetector(
                status_change_threshold=config["status_change_threshold"],
                red_threshold_lower1=config["red_threshold_lower1"],
                red_threshold_upper1=config["red_threshold_upper1"],
                red_threshold_lower2=config["red_threshold_lower2"],
                red_threshold_upper2=config["red_threshold_upper2"],
                min_contour_area=config["min_contour_area"],
                morphological_kernel_size=config["morphological_kernel_size"],
                dilation_iterations=config["dilation_iterations"],
                symmetric_distance_threshold=config["symmetric_distance_threshold"]
            )

            # Detect taillights
            bounding_boxes, brake_status, brightness = detector.detect_taillights(image, False)

            # Annotate image
            result = image.copy()
            for box in bounding_boxes:
                xmin, ymin, xmax, ymax = box
                color = (0, 0, 255) if brake_status else (0, 255, 0)  # Red if braking, green otherwise
                cv2.rectangle(result, (xmin, ymin), (xmax, ymax), color, 2)

            # Add text indicating brake status and configuration name
            status_text = f"{config['name']}: {'BRAKE ON' if brake_status else 'BRAKE OFF'}"
            cv2.putText(result, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if brake_status else (0, 255, 0), 2)

            # Save result if output directory is specified
            if output_dir:
                output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_{config['name']}.jpg")
                cv2.imwrite(output_path, result)

            # Display result
            if show_results:
                plt.subplot(len(PARAM_CONFIGS), 1, i + 1)
                plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                plt.title(status_text)
                plt.axis('off')

        if show_results:
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_comparison.jpg"))
            plt.show()


def test_detector_on_sequence(image_dir, class_name, output_dir=None, show_results=True):
    """
    Test the taillight detector on a sequence of images from a specific class

    Args:
        image_dir: Directory containing sequence images
        class_name: Class name for sequence labeling (e.g., 'BOO' for braking)
        output_dir: Directory to save output images (optional)
        show_results: Whether to display visualization
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get list of image files in sequence
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not image_files:
        print(f"No image files found in {image_dir}")
        return

    print(f"Found {len(image_files)} images in sequence")

    # Create detector with default parameters
    detector = TaillightDetector()

    # Process each image in sequence
    brake_status_history = []
    brightness_history = []

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        # Detect taillights
        bounding_boxes, brake_status, brightness = detector.detect_taillights(image, False)
        brake_status_history.append(brake_status)
        brightness_history.append(brightness)

        # Annotate image
        result = image.copy()
        for box in bounding_boxes:
            xmin, ymin, xmax, ymax = box
            color = (0, 0, 255) if brake_status else (0, 255, 0)  # Red if braking, green otherwise
            cv2.rectangle(result, (xmin, ymin), (xmax, ymax), color, 2)

        # Add frame number and status
        cv2.putText(result, f"Frame {i} - {brake_status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if brake_status else (0, 255, 0), 2)

        # Save result if output directory is specified
        if output_dir:
            output_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(output_path, result)

        # Display result
        if show_results and i % 10 == 0:  # Show only every 10th frame to avoid too many windows
            cv2.imshow(f"Frame {i}", result)
            cv2.waitKey(100)

    # Close all windows
    cv2.destroyAllWindows()

    # Calculate accuracy (assuming class name indicates ground truth)
    expected_braking = 'B' in class_name[0]  # First letter is 'B' for braking
    accuracy = sum(1 for status in brake_status_history if status == expected_braking) / len(brake_status_history)

    print(f"Sequence class: {class_name}")
    print(f"Expected braking: {expected_braking}")
    print(f"Detection accuracy: {accuracy:.2f}")

    # Plot brightness over time
    plt.figure(figsize=(10, 5))
    plt.plot(brightness_history)
    plt.axhline(y=detector.status_change_threshold, color='r', linestyle='--')
    plt.title(f"Brightness over time for sequence {class_name}")
    plt.xlabel("Frame")
    plt.ylabel("Brightness")
    plt.grid(True)

    if output_dir:
        plt.savefig(os.path.join(output_dir, "brightness_plot.jpg"))

    if show_results:
        plt.show()


def main():
    # Example usage for testing on individual images
    image_dir = "data/test/"
    output_dir = "results/individual"

    test_detector_on_images(image_dir, output_dir, show_results=True)

    # Example usage for testing on a sequence
    sequence_dir = "dataset/rear_signal_dataset/20160805_g1k17-08-05-2016_15-57-59_idx99/20160805_g1k17-08-05-2016_15-57-59_idx99_BOO/20160805_g1k17-08-05-2016_15-57-59_idx99_BOO_00002671/light_mask"
    output_dir = "results/sequence"

    test_detector_on_sequence(sequence_dir, "BOO", output_dir, show_results=True)


if __name__ == "__main__":
    main()
