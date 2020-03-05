import cv2
import numpy as np


class Image:
    def __init__(self, rgb_image):
        self.rgb = rgb_image

    @property
    def shape(self):
        return self.rgb.shape

    @property
    def rgb(self):
        return self._rgb

    @rgb.setter
    def rgb(self, rgb_img):
        self._rgb = rgb_img

    @property
    def lab(self):
        if self.rgb.shape[-1] == 1:
            img = cv2.cvtColor(self.rgb, cv2.COLOR_GRAY2RGB)
            return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        return cv2.cvtColor(self.rgb, cv2.COLOR_RGB2LAB)

    @property
    def rgb_split(self):
        return cv2.split(self.rgb)

    @property
    def lab_split(self):
        return cv2.split(self.lab)

    def show(self):
        bgr = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("Image", bgr)
        cv2.waitKey()

    def encode(self, path):
        bgr = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)


def decode(path):
    with open(path, "rb") as f:
        img = cv2.imdecode(
            np.frombuffer(f.read(), np.uint8),
            cv2.IMREAD_UNCHANGED
        )
    shape = img.shape
    if len(shape) == 2:
        return Image(img[..., np.newaxis])
    elif shape[-1] == 3:
        return Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError()

