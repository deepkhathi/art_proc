# General utilities for logging, plots, etc
import logging
from typing import Optional, Union

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from numpy import ndarray


def configure_logging(
    level: int = logging.INFO,
    filename: Optional[str] = None,
    filemode: str = "w+",
    **kwargs
):
    r"""Set logging configurations

    Parameters
    ----------
    level : int, default=logging.INFO
    filename : str, optional
        If None, logs to console
    filemode : str, default="w+"
    **kwargs
        Additional configurations for logging.basicConfig
    """
    logging.basicConfig(
        level=level,
        filename=filename,
        filemode=filemode,
        **kwargs)


def init_logger(name: Optional[str] = None) -> logging.Logger:
    r"""Initialize standard logger with provided name (default="root")"""
    if not logging.getLogger().handlers:
        configure_logging(format=r"%(name)s :: %(levelname)-4s :: %(message)s")
    return logging.getLogger(name)


def display_image(
    image: ndarray,
    cmap: Union[bool, str] = False,
    ax: Optional[Axes] = None
) -> None:
    r"""Display image using `matplotlib.pyplot.imshow`

    Parameters
    ----------
    image : numpy.ndarray
    cmap : Union[bool, str], default=False
        Colormap; if `True`, this uses `gray`.
        If `str`, must be a valid `seaborn` or `matplotlib` colormap
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on; if None, create new Axes object
    """
    draw = False
    if isinstance(cmap, bool) and cmap:
        cmap = "gray"
    elif not cmap:
        cmap = None
    if ax is None:
        draw = True
        _, ax = plt.subplots()
    ax.imshow(image, cmap=cmap)
    sns.despine(ax=ax, left=True, bottom=True)
    ax.tick_params(
        axis="both",
        labelbottom=False,
        labelleft=False)
    ax.axis("off")
    if draw:
        plt.show()
    return None
