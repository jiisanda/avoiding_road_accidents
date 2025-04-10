import enum
import numpy as np

# In seconds
FRAME_LIFETIME = 2000


class CarStatus(enum.Enum):
    UNKNOWN = 0
    NOT_BRAKING = 1
    BRAKING = 2


class DetectedCar:
    id = 0
    frames = []
    status = CarStatus.UNKNOWN

    def __init__(self, car_id, first_frame):
        self.id = car_id
        self.frames = [first_frame]

    def get_id(self):
        return self.id

    def set_status(self, new_status):
        self.status = new_status

    def get_status(self):
        return self.status

    def add_frame(self, frame):
        self.frames.append(frame)

    def get_last_frame(self):
        return self.frames[len(self.frames) - 1]

    def get_first_frame(self):
        return self.frames[0]

    def get_all_frames(self):
        return self.frames

    def remove_old_frames(self, current_time):
        cleaned_list = [frame for frame in self.frames if frame.get_time() + FRAME_LIFETIME >= current_time]
        self.frames = cleaned_list

    def get_frame_from_past(self, current_time, time_ago):
        # Reverse loop
        for i in range(len(self.frames) - 1, -1, -1):
            if self.frames[i].get_time() < current_time - time_ago:
                return self.frames[i]

    def get_average_brightness(self, current_time, time_boundary_min, time_boundary_max):
        brightness = []

        for i in range(len(self.frames) - 1, -1, -1):
            if self.frames[i].get_time() < current_time - time_boundary_max:
                print(f"Find brightness:{brightness}")
                #print("QUIT")
                return np.median(brightness)

            if self.frames[i].get_time() < current_time - time_boundary_min:
                #print("Append")
                brightness.append(self.frames[i].get_brightness())

        if len(brightness) == 0:
            return np.nan

        print(f"Find brightness:{brightness}")
        return np.median(brightness)


