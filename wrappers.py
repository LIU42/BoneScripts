class BoneScript:
    def __init__(self, x, y, w, h, code):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.code = code

    def __str__(self):
        return f'box: [{self.x}, {self.y}, {self.w}, {self.h}] code: {self.code}'
    
    @property
    def x1(self):
        return self.x

    @property
    def y1(self):
        return self.y

    @property
    def x2(self):
        return self.x + self.w

    @property
    def y2(self):
        return self.y + self.h

    @staticmethod
    def from_box(box):
        return BoneScript(box[0], box[1], box[2], box[3], None)
