import math as math
import cv2

from vehicle.car_frame import CarFrame
from vehicle.detected_car import DetectedCar

boxes_distance_threshold = 200

# TODO: USE IT !
boxes_size_threshold = [100, 100]

# After what time remove undetectable car from list
# In milliseconds
undetectableCarTimeToLive = 500


class FramesStorage:
    detected_cars = []
    last_car_index = 0

    def __init__(self):
        self.detected_cars = []

    def add_new_car_to_list(self, car_frame):
        car_id = self.last_car_index
        self.last_car_index += 1

        detectedCar = DetectedCar(car_id, car_frame)

        self.detected_cars.append(detectedCar)

        print(f"Find new car! Id: {car_id}")

        return detectedCar

    def clear_old_frames(self, current_time):
        for car in self.detected_cars:
            car.remove_old_frames(current_time)

    def clear_long_time_undetectable_cars(self, current_time):

        # Reverse loop
        for i in range(len(self.detected_cars), 0, -1):
            cur_ind = i - 1

            car = self.detected_cars[cur_ind]
            lastCarFrame = car.get_last_frame()

            if lastCarFrame.get_time() + undetectableCarTimeToLive < current_time:
                print(f"Remove car {car.get_id()}; CurrentTime:{current_time}, LastDetected: {lastCarFrame.get_time()}")

                self.detected_cars.pop(cur_ind)

    def get_car(self, time, bounding_box, crop_img):
        car_frame = CarFrame(time, bounding_box, crop_img)
        new_box_center = [bounding_box[0] + int(bounding_box[2] / 2), bounding_box[1] + int(bounding_box[3] / 2)]
        new_box_size = [bounding_box[2], bounding_box[3]]

        if len(self.detected_cars) > 0:

            near_detected_car = None
            near_car_distance = 999

            for i in range(len(self.detected_cars)):
                car = self.detected_cars[i]

                # Bounding box size test
                last_frame_size = car.get_last_frame().get_size()

                if abs(new_box_size[0] - last_frame_size[0]) > boxes_size_threshold[0] or abs(new_box_size[1] - last_frame_size[1]) > boxes_size_threshold[1]:
                    #print("Bounding box fail")
                    # Bounding box size check fail
                    continue

                # Centroid check
                last_frame_centroid = car.get_last_frame().get_centroid()
                distance_to_center = math.dist(last_frame_centroid, new_box_center)

                if distance_to_center < near_car_distance:
                    near_detected_car = car
                    near_car_distance = distance_to_center

            # TODO: Also check that this car is not used already!
            if near_car_distance < boxes_distance_threshold:
                detected_car = near_detected_car
                near_detected_car.AddFrame(car_frame)
            else:
                detected_car = self.add_new_car_to_list(car_frame)
        else:
            detected_car = self.add_new_car_to_list(car_frame)

        return detected_car

    # TODO: Move it to Detected car method
    def get_car_path(self, car_id):
        car = next((car for car in self.detected_cars if car.get_id() == car_id), None)

        all_car_frames = car.GetAllFrames()
        all_centroids = []

        for frame in all_car_frames:
            all_centroids.append(frame.get_centroid())

        return all_centroids


