from abc import ABC, abstractmethod
import numpy as np


class Similarity(ABC):
    @abstractmethod
    def __call__(self, img1, img2):
        pass


class SSIM(Similarity):
    def __init__(self, k1: float = 0.01, k2: float = 0.03, l: int = 255):
        self._k1 = k1
        self._k2 = k2
        self._L = l

    def __call__(self, img1, img2):
        assert len(img1.shape) == 2 and len(img2.shape) == 2
        assert img1.shape == img2.shape
        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1 = np.sqrt(((img1 - mu1) ** 2).mean())
        sigma2 = np.sqrt(((img2 - mu2) ** 2).mean())
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

        C1 = (self._k1 * self._L) ** 2
        C2 = (self._k2 * self._L) ** 2
        C3 = C2 / 2
        l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
        c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
        s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
        ssim = l12 * c12 * s12
        return ssim


class PCC(Similarity):
    def __call__(self, img1, img2):
        pass


