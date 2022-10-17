class Ellipse:
    def __init__(self, label, x, y, width, height):
        self.x = int(x)
        self.y = int(y)
        self.r_width = int(width)
        self.r_height = int(height)
        self.label = label

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_r_width(self):
        return self.r_width

    def get_r_height(self):
        return self.r_height

    def get_label(self):
        return self.label