import numpy as np
import cv2


def tenengrad_focus_value(gray_img):
    """Implements the Tenengrad focus measure operator.
    Based on the gradient of the image.

    Args:
        gray_img: the gray image with shape (w, h) to apply the measure

    Returns:
        the degree of focus
    """
    assert len(gray_img.shape) == 2
    gaussian_x = cv2.Sobel(np.float32(gray_img), cv2.CV_64F, 1, 0)
    gaussian_y = cv2.Sobel(np.float32(gray_img), cv2.CV_64F, 0, 1)
    return np.mean(gaussian_x * gaussian_x +
                   gaussian_y * gaussian_y)


class BlurDetector:
    def __init__(self, threshold=0.65, kernel=(7, 7)):
        """Init a blur detector

        Args:
            threshold: the value of blur level [0,1] to separate the blur and sharp image
            kernel: Gaussian blur kernel to blur image.
        """
        self._thd = threshold
        self._blur_kernel = kernel

    def detect(self, gray_img):
        """Detect if a image is blur.

        Args:
            gray_img: image to measure if it is blur.

        Returns:
            bool, return True if image is blur.
        """
        low_pass = cv2.GaussianBlur(gray_img, self._blur_kernel, 0)
        ratio = tenengrad_focus_value(low_pass)/tenengrad_focus_value(gray_img)
        return ratio > self._thd
