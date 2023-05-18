# Base class

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray, rgba2rgb

from .utils import init_logger


class BaseImage():
    r"""Base class - read and convert grayscale image

    Parameters
    ----------
    rgb_image : numpy.ndarray
        RGB(A) image (either (N, M, 3) or (N, M, 4))
    class_name : str, optional
        Class name for logger. If None, defaults to root
    verbose : bool, default=False

    Attributes
    ----------
    rgb_image : numpy.ndarray
        RGB(A) image
    gray_image : numpy.ndarray, shape=(N, M)
    verbose : bool, default=False
    class_name : str
    logger : logging.Logger
    """

    def __init__(
        self,
        rgb_image: np.ndarray,
        class_name: Optional[str] = None,
        verbose: bool = False
    ):
        self.verbose = verbose
        self.logger = init_logger(name=class_name)
        self.rgb_image = rgb_image
        self._orig_gray = False
        im_shape = self.rgb_image.shape
        if len(im_shape) == 2:
            self.gray_image: np.ndarray = self.rgb_image.copy()
            self._orig_gray = True
        elif len(im_shape) == 3 and im_shape[-1] == 4:
            self.rgb_image = rgba2rgb(self.rgb_image)
        elif len(im_shape) != 3:
            raise ValueError(
                f"Invalid shape {im_shape} must be (N, M, 3) "
                f"or (N, M, 4) or (N, M)")
        if not self._orig_gray:
            self.gray_image: np.ndarray = rgb2gray(self.gray_image)

    def _return_image(self, image: np.ndarray, return_flag: bool):
        return image if return_flag else None

    def _run_show(self):
        if self.verbose:
            plt.show()
        plt.close()
