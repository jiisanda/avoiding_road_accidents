import cv2
import os
import time
import glob
from yolo.yolo8_detector import YOLOv8Detector


def main():
    detector = YOLOv8Detector(model_path='models/yolov8m.pt')

    input_dir = 'dataset/images'
    output_dir = 'results'

    os.makedirs(output_dir, exist_ok=True)

    image_paths = glob.glob(os.path.join(input_dir, '*.jpg')) + \
                  glob.glob(os.path.join(input_dir, '*.png'))

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images")

    total_time = 0
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to read image: {image_path}")
            continue

        start_time = time.time()
        processed_image = detector.process_image(image)
        end_time = time.time()

        process_time = end_time - start_time
        total_time += process_time

        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, processed_image)

        print(f"Processed {i + 1}/{len(image_paths)}: {filename} in {process_time:.3f} seconds")

    print(f"All images processed. Average time: {total_time / len(image_paths):.3f} seconds per image")


if __name__ == "__main__":
    main()