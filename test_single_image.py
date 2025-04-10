import cv2
import os
import time
from yolo.yolo8_detector import YOLOv8Detector


def main():
    detector = YOLOv8Detector(model_path='models/yolov8m.pt')

    image_path = 'data/frame00017771.png'

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    start_time = time.time()
    processed_image = detector.process_image(image)
    end_time = time.time()

    print(f"Processing time: {end_time - start_time:.3f} seconds")

    cv2.imshow("Result", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output_path = 'output.jpg'
    cv2.imwrite(output_path, processed_image)
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()
