import math
import cv2
class BoundingBox:

    def __init__(self, label, x, y, width, height, rotation=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.rotation = rotation

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_rotation(self):
        return self.rotation

    def get_label(self):
        return self.label

    def set_label(self, label):
        self.label = label

    def get_xywh(self):
        return self.x, self.y, self.width, self.height

    # Assuming rotation is 0, ToDo: add rotation
    def get_lefttop_rightbottom(self):
<<<<<<< HEAD
        x2 = self.x + self.width
        y2 = self.y + self.height
        return self.x, self.y, x2, y2
=======
        return self.x, self.y, self.x + self.width, self.y + self.height
>>>>>>> ellipse_annotation

    # Assuming rotation is 0, ToDo: add rotation
    def set_lefttop_rightbottom(self, x1, y1, x2, y2):
        self.width = x2 - x1
        self.height = y2 - y1
        self.x = x1
        self.y = y1