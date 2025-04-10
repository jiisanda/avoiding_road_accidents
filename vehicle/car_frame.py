class CarFrame:
    time = 0

    # Format - [start_y, start_x, height, width]
    bounding_box = []
    centroid = []

    image = []
    light_brightness = 0

    def __init__(self, frame_time, bounding_box, image):
        self.time = frame_time
        self.bounding_box = bounding_box

        center_y = bounding_box[0] + int(bounding_box[2] / 2)
        center_x = bounding_box[1] + int(bounding_box[3] / 2)

        self.centroid = [center_y, center_x]
        self.image = image

    def get_time(self):
        return self.time

    def get_bounding_box(self):
        return self.bounding_box

    def get_centroid(self):
        return self.centroid

    def get_size(self):
        return [self.bounding_box[2], self.bounding_box[3]]

    def get_image(self):
        return self.image

    def set_brightness(self, brightness):
        self.light_brightness = brightness

    def get_brightness(self):
        return self.light_brightness
