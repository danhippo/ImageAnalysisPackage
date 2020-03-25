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


# if __name__ == '__main__':
#     import pathlib
#     import shutil
#     from Image import Image
#     # root = pathlib.Path(r"D:\RD_Data\KS\AI_Competation\Merge_data\PPC2_merge\defect")
#     root = pathlib.Path(
#         r"D:\RD_Data\KS\AI_Competation\train_set\KS_AICompetition01\front_green\normal")
#     # root = pathlib.Path(r"D:\RD_Data\KS\AI_Competation\Merge_data\PPC4_merge\defect")
#     # root = pathlib.Path(r"D:\RD_Data\KS\AI_Competation\Merge_data\PPC4_merge\normal")
#     # root = pathlib.Path(
#     #     r"D:\RD_Data\KS\AI_Competation\train_set\KS_AICompetition01\back_gold\defect")
#
#     dst_dir = pathlib.Path(r"D:\test\Blur_detect")
#     if dst_dir.exists():
#         shutil.rmtree(str(dst_dir))
#     blur_dir = dst_dir.joinpath("blur")
#     sharp_dir = dst_dir.joinpath("sharp")
#     blur_dir.mkdir(exist_ok=True, parents=True)
#     sharp_dir.mkdir(exist_ok=True, parents=True)
#     imgs = root.glob("*.bmp")
#
#     blur_detector = BlurDetector(0.65)
#     for img_path in imgs:
#         try:
#             ori_img = Image.from_file(str(img_path))
#             if blur_detector.detect(gray_img=ori_img.gray):
#                 shutil.copyfile(str(img_path), blur_dir.joinpath(img_path.name))
#             else:
#                 shutil.copyfile(str(img_path), sharp_dir.joinpath(img_path.name))
#         except Exception:
#             print(img_path.name)
