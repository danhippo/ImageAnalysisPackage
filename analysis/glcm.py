import math
import numpy as np
from enum import Enum


class GLCM:
    def __init__(self, glcm: np.array):
        """Init a gray level co-occurence matrix.

        Args:
            glcm: A numpy array with size of (grey level)*(grey level).
        """
        assert len(glcm.shape) == 2
        assert glcm.shape[0] == glcm.shape[1]
        assert 1 <= len(glcm) <= 256
        self._gray_level = glcm.shape[0]
        self._glcm = glcm

    @property
    def glcm(self):
        """The input glcm

        Returns: glcm

        """
        return self._glcm

    @property
    def asm(self):
        """A measure of homogeneity of the image.

        Returns: Angular second-moment feature

        """
        return np.sum(self._nglcm * self._nglcm)

    @property
    def contrast(self):
        """A measure of the contrast or the amount of local variations present in an image.

        Returns: Contrast feature.

        """
        gray_level_diff = np.array([(i - j) for i in range(self.gray_level) for j in
                                    range(self.gray_level)])
        gray_level_diff = np.reshape(gray_level_diff, (self.gray_level, self.gray_level))
        weights = gray_level_diff * gray_level_diff
        return np.sum(weights * self._nglcm)

    @property
    def entropy(self):
        """A measure of randomness of the GLCM

        Returns: Entropy feature.

        """
        mask_nglcm = self._nglcm
        mask_nglcm[mask_nglcm == 0.0] = 1.0
        return np.sum(- mask_nglcm * np.log(mask_nglcm))

    @property
    def idm(self):
        """Large if big values are on the main diagonal

        Returns: Inverse differential moment feature

        """
        gray_level_diff = np.array([(i - j) for i in range(self.gray_level) for j in
                                    range(self.gray_level)])
        gray_level_diff = np.reshape(gray_level_diff, (self.gray_level, self.gray_level))
        weights = gray_level_diff * gray_level_diff
        return np.sum(self._nglcm / (1 + weights))

    @property
    def gray_level(self):
        """gray level of GLCM

        Returns: gray level

        """
        return self._gray_level

    @property
    def _nglcm(self):
        return self._glcm / np.sum(self._glcm)


class GLCMAngle(Enum):
    Ang_0 = 0
    Ang_45 = 45
    Ang_90 = 90
    Ang_135 = 135


class GLCMComputer:
    def __init__(self, gray_level: int = 256):
        """Init a GLCM computer

        Args:
            gray_level: the gray level of expected glcm
        """
        if not 0 < gray_level <= 256:
            raise ValueError("Expected gray level in the range of (1, 256)")
        self._gray_level = gray_level
        self._supported_angle = [GLCMAngle.Ang_0,
                                 GLCMAngle.Ang_45,
                                 GLCMAngle.Ang_90,
                                 GLCMAngle.Ang_135]

    def __call__(self,
                 gray_img: np.array,
                 angles: list = None
                 ) -> dict:
        """Function to get a GLCM from a gray image on specific angles.

        Args:
            gray_img: A gray image to compute GLCM.
            angles:
                A list of GLCMAngle Enum.
                Default value is None, and it will compute GLCMs of all angles.

        Returns:
            A dictionary of {GLCMAngle: GLCM}
        """
        if angles is None:
            angle_set = set(self._supported_angle)
        else:
            angle_set = set(angles)

        if not angle_set.issubset(self._supported_angle):
            raise ValueError("Unsupported input angle:{}".format(angle_set-set(self._supported_angle)))

        if not len(gray_img):
            raise ValueError()

        (height, width) = gray_img.shape

        srcdata = gray_img
        max_gray_level = np.max(gray_img)
        if max_gray_level >= self._gray_level:
            srcdata = gray_img.astype(np.float) * self._gray_level / max_gray_level - 1
            srcdata = np.clip(srcdata, 0, self._gray_level)
            srcdata = srcdata.astype(np.uint8)

        ret = {}
        for angle in angle_set:
            glcm = np.zeros((self._gray_level, self._gray_level), dtype=np.float)
            d_y = round(math.sin(-(math.pi / 180) * angle.value))
            d_x = round(math.cos(-(math.pi / 180) * angle.value))

            for j in range(height):
                for i in range(width):
                    if (j + d_y) in range(height) and (i + d_x) in range(width):
                        row = srcdata[j][i]
                        col = srcdata[j + d_y][i + d_x]
                        glcm[row][col] += 1.0

                    if (j - d_y) in range(height) and (i - d_x) in range(width):
                        row = srcdata[j][i]
                        col = srcdata[j - d_y][i - d_x]
                        glcm[row][col] += 1.0
            ret.update({angle: GLCM(glcm)})
        return ret
