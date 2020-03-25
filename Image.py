import cv2
import numpy as np


class Image:
    def __init__(self, rgb_image):
        """

        Args:
            rgb_image:
        """
        self._rgb = rgb_image
        self._dft = cv2.dft(np.float32(self.gray), flags=cv2.DFT_COMPLEX_OUTPUT)

    @staticmethod
    def from_file(path):
        """Init a Image object with image path.

        Args:
            path(str): image path

        Returns:
            an Image instance
        """
        with open(path, "rb") as f:
            img = cv2.imdecode(
                np.frombuffer(f.read(), np.uint8),
                cv2.IMREAD_COLOR
            )
        return Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    @staticmethod
    def from_numpy(src_img: np.array):
        """Init a Image object from np.array.

        Args:
            src_img(np.array): a 2-D gray image or 3-D rgb image.

        Returns:
            an Image instance
        """
        if len(src_img.shape) == 2:
            src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2RGB)
        return Image(src_img)

    @property
    def shape(self):
        """The shape (w, h) of image"""
        return self.rgb.shape[0], self.rgb.shape[1]

    @property
    def rgb(self):
        """The image in RGB color space"""
        return self._rgb

    @property
    def lab(self):
        """The image in CIE-Lab color space"""
        return cv2.cvtColor(self.rgb, cv2.COLOR_RGB2LAB)

    @property
    def gray(self):
        """The image in form of gray"""
        return cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)

    def show(self):
        """Show the image"""
        bgr = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("Image", bgr)
        cv2.waitKey()

    @property
    def dft(self):
        """The discrete fourier transform result on gray image"""
        return self._dft

    @property
    def dft_spectrum(self):
        """The dft spectrum on gray image"""
        dft_shift = np.fft.fftshift(self.dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        return magnitude_spectrum

    @property
    def fft_focus_value(self):
        """The focus value of image via fourier transform result"""
        R = self.dft[:, :, 0]
        I = self.dft[:, :, 1]
        element = np.sqrt((np.power(R, 2) + np.power(I, 2))*np.abs(np.arctan2(I, R)))
        S = np.sum(element)/(element.shape[0]*element.shape[1])
        return S

    @property
    def tenengrad_focus_value(self):
        """The Tenengrad focus value"""
        gaussian_x = cv2.Sobel(np.float32(self.gray), cv2.CV_64F, 1, 0)
        gaussian_y = cv2.Sobel(np.float32(self.gray), cv2.CV_64F, 0, 1)
        return np.mean(gaussian_x * gaussian_x +
                       gaussian_y * gaussian_y)
