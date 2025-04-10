from torch.serialization import add_safe_globals
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

from ultralytics.nn.tasks import DetectionModel

from vehicle.frame_storage import FramesStorage


class YOLOv8Detector:
    def __init__(self, model_path='models/yolov8m.pt', conf_thresh=0.5, iou_thresh=0.3):
        """
        Initialize YOLOv8 detector

        Args:
            model_path: Path to YOLOv8 model weights
            conf_thresh: Confidence threshold for detections
            iou_thresh: IoU threshold for NMS
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.frame_storage = FramesStorage()

        # Load model
        add_safe_globals([DetectionModel])
        self.model = YOLO(model_path)

        # Class filter (only detect vehicles)
        # COCO classes: 2-car, 5-bus, 7-truck
        self.vehicle_classes = [2, 5, 7]

        # Generate colors for visualization
        self.color_palette = self.generate_colors(30)

    def generate_colors(self, num):
        """Generate random colors for visualization"""
        r = lambda: np.random.randint(64, 255)
        return [(r(), r(), r()) for _ in range(num)]

    def get_color(self, id):
        """Get color for a specific ID"""
        return np.array(self.color_palette[id % len(self.color_palette)])

    def process_image(self, image, timestamp=None):
        """
        Process an image with YOLOv8 and detect taillights

        Args:
            image: Input image
            timestamp: Optional timestamp (for video processing)

        Returns:
            Processed image with detections
        """
        if timestamp is None:
            timestamp = time.time() * 1000  # Convert to milliseconds

        # Run YOLOv8 inference
        results = self.model(image, conf=self.conf_thresh, iou=self.iou_thresh, verbose=False)

        # Process results
        for result in results:
            boxes = result.boxes

            # Process each detection
            for box in boxes:
                # Get box details
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                # Filter for vehicle classes
                if class_id in self.vehicle_classes:
                    # Convert to [x, y, w, h] format
                    w = x2 - x1
                    h = y2 - y1
                    bbox = [x1, y1, w, h]

                    # Process bounding box
                    self.process_bounding_box(image, bbox, timestamp)

        return image

    def process_bounding_box(self, image, bbox, timestamp):
        """Process a vehicle bounding box and detect taillights"""
        x, y, w, h = bbox
        if x <= 0 or y <= 0 or w <= 0 or h <= 0 or y + h > image.shape[0] or x + w > image.shape[1]:
            return

        # Crop vehicle image
        cropped_img = image[y:y + h, x:x + w].copy()

        # Cleanup old data
        self.frame_storage.clear_long_time_undetectable_cars(timestamp)
        self.frame_storage.clear_old_frames(timestamp)

        # Get or create car object
        current_car = self.frame_storage.get_car(timestamp, bbox, cropped_img)
        car_id = current_car.get_id()

        # Process taillights
        from vehicle.tail_detector import analyze_car_status
        rects = analyze_car_status(timestamp, current_car)

        if len(rects) == 0:
            return

        # Determine color based on braking status
        from vehicle.detected_car import CarStatus
        if current_car.get_status() == CarStatus.BRAKING:
            color = (0, 0, 255)  # Red for braking
        elif current_car.get_status() == CarStatus.NOT_BRAKING:
            color = (0, 0, 128)  # Dark red for not braking
        else:
            color = (0, 0, 0)  # Black for unknown

        # Draw taillight detections
        for rect in rects:
            cv2.rectangle(image,
                          (x + rect[0], y + rect[1]),
                          (x + rect[2], y + rect[3]),
                          color, 2)

        # Draw vehicle bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (52, 64, 235), 1)
        cv2.putText(image, f"Id: {car_id}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 235), 2)
        cv2.putText(image, f"{current_car.get_status().name}", (x + 50, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 235), 2)

        return image
