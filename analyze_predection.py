import os
import cv2
import numpy as np
from ultralytics import YOLO


def analyze_predictions():
    # Load the model
    model = YOLO('runs/detect/train5/weights/best.pt')

    # Class names mapping
    class_names = {
        0: 'OOO',
        1: 'BOO',
        2: 'OLO',
        3: 'BLO',
        4: 'OOR',
        5: 'BOR',
        6: 'BLR'
    }

    def draw_detections(image, detections):
        # Sort detections by confidence (highest first)
        detections.sort(key=lambda x: x[3], reverse=True)

        for box, class_id, class_name, confidence in detections:
            x1, y1, x2, y2 = map(int, box)

            # Draw green rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Prepare text
            text = f'{class_name} {confidence:.2f}'

            # Text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1

            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

            # Calculate text position - top left corner
            text_x = x1
            text_y = y1 - 5  # 5 pixels above the box

            # If text would go above image top, put it inside the box at the top
            if text_y - text_height < 0:
                text_y = y1 + text_height + 5

            # Draw black background for text
            cv2.rectangle(image,
                          (text_x, text_y - text_height - 5),
                          (text_x + text_width + 5, text_y + 5),
                          (0, 0, 0),
                          -1)

            # Draw text
            cv2.putText(image,
                        text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        (255, 255, 255),
                        font_thickness,
                        cv2.LINE_AA)

        return image

    # Directory with test images
    test_dir = "data/yolo_dataset/images/val"
    output_dir = "runs/detect/predictions"
    os.makedirs(output_dir, exist_ok=True)

    # Process images
    for img_file in os.listdir(test_dir):
        if img_file.endswith('.png') or img_file.endswith('.jpg'):
            img_path = os.path.join(test_dir, img_file)

            # Perform prediction
            results = model(img_path)

            # Load image for visualization
            img = cv2.imread(img_path)

            # Collect all detections
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get prediction details
                    x1, y1, x2, y2 = box.xyxy[0]
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = class_names.get(class_id, str(class_id))

                    detections.append(([x1, y1, x2, y2], class_id, class_name, confidence))

            # Draw all detections sorted by confidence
            img = draw_detections(img, detections)

            # Save the annotated image
            output_path = os.path.join(output_dir, img_file)
            cv2.imwrite(output_path, img)

            # Display result
            cv2.imshow('Detection', img)
            key = cv2.waitKey(0)

            # Press 'q' to quit, any other key to continue
            if key == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    analyze_predictions()