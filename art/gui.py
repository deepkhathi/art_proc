# GUI (equivalent to argparse) using GooeyParser

import os
import sys
from typing import Tuple

import numpy as np
from gooey import Gooey, GooeyParser
from matplotlib.pyplot import imsave
from skimage.io import imread

sys.path.append(os.getcwd())  # noqa
from art.boundary import BoundaryImage  # noqa
from art.utils import display_image, init_logger  # noqa


class GUI():
    r"""Maintain GUI parsing and workflow

    Parameters
    ----------
    args : Any
        `GooeyParser.parse_args` contents

    Attributes
    ----------
    args : Any
        Raw parsed arguments
    options_mapper : Dict[str, str]
        Mapping option text to function for `art.boundary.BoundaryImage`
    """

    def __init__(self, args):
        self.options_mapper = {
            "Gamma": "gamma",
            "Sigmoid": "sigmoid",
            "CLAHE": "clahe",
            "Histogram Equalization": "histeq",
            "Contrast Stretching": "contrast_stretch"}
        self.logger = init_logger("DefaultGUI")
        self.args = args
        self.parse_args()
        self.load_image()
        self.boundary_handler = BoundaryImage(
            self.rgb_image,
            photo_correction=self.is_photo,
            photo_corr_method=self.exposure_algo)

    def load_image(self):
        r"""Load RGB(A) image"""
        self.rgb_image: np.ndarray = imread(self.filename)

    def parse_args(self) -> None:
        r"""Parse and save `args` contents"""
        # defaults
        self.filename: str = self.args.filename
        self.output_dir: str = self.args.output_dir
        # image settings
        self.is_photo: bool = self.args.is_photo
        self.exposure_algo: str = self.options_mapper.get(
            self.args.exposure_algo, "Sigmoid")
        # output settings
        self.n_classes: int = self.args.n_classes
        self.output_ftype: str = self.args.filetype
        self.transp_bg: bool = self.args.transp_bg
        self.invert: bool = self.args.invert
        self.apply_post: bool = self.args.post_filter
        self.resolution: int = self.args.resolution
        self.resize_settings: Tuple[int, int] = (
            self.args.width * self.resolution,
            self.args.height * self.resolution)
        if self.args.width <= 0 or self.args.height <= 0:
            self.resize_settings = None
        self.logger.info(self.args)

    def run(self):
        boundary_image = self.boundary_handler.run_pipeline(
            n_classes=self.n_classes,
            apply_post_filter=self.apply_post,
            resize_shape=self.resize_settings,
            invert_colors=self.invert,
            transparent_background=self.transp_bg)
        # display_image(boundary_image)
        # self.boundary_handler._run_show()
        fname = os.path.basename(self.filename.rsplit('.', 1)[0])
        output_filename = f"{fname}_boundary_mclass_{self.n_classes}"
        if self.resize_settings is not None:
            output_filename += f"__resized"
        output_loc = os.path.join(
            self.output_dir, f"{output_filename}.{self.output_ftype.lower()}")
        imsave(output_loc, boundary_image, cmap="gray")
        self.logger.info(f"Boundary image saved to {output_loc}")
        return None


def parser() -> GooeyParser:
    r"""GUI parser with inputs similar to ArgumentParser"""
    parser = GooeyParser()
    step1 = parser.add_argument_group("1. Select File and Output Location")
    step1.add_argument("filename", help="Image to process", metavar="Filename",
                       widget="FileChooser")
    step1.add_argument("output_dir", metavar="Output Directory",
                       widget="DirChooser")
    im_settings = parser.add_argument_group(
        "2. Input Image Settings",
        "Specify the input image settings (photo, exposure correction method)")
    im_settings.add_argument("--is_photo", metavar="Is the image a photo?",
                             widget="CheckBox", action="store_true")
    im_settings.add_argument(
        "--exposure_algo", metavar="Exposure correction algorithm",
        help="Specify exposure correction algorithm to reduce the "
             "effect of bright spots on thresholding",
        default="Sigmoid",
        widget="Dropdown",
        choices=["Gamma", "Sigmoid", "Log",
                 "CLAHE",
                 "Histogram Equalization", "Contrast Stretching"])
    out_settings = parser.add_argument_group(
        "3. Output Settings",
        "Specify output image settings (filetype, background transparency, "
        "number of thresholds,\nfinal color inversion)")
    out_settings.add_argument("--n_classes", metavar="Number of thresholds",
                              help="Number of thresholds is linked to how "
                              "much detail is picked up (min. 2).\n"
                              "Recommended: 2 for borders, 3-4 if detail is important",
                              action="store", widget="Slider",
                              gooey_options={"min": 2, "max": 6},
                              default=2, type=int)
    out_settings.add_argument("--filetype", metavar="Output filetype",
                              widget="Dropdown", default="PNG",
                              choices=["PNG", "JPG", "PDF"])
    out_settings.add_argument("--transp_bg", metavar="Set background as transparent",
                              help="This is mainly relevant to PDF images",
                              action="store_true", widget="CheckBox")
    out_settings.add_argument("--invert", metavar="Invert output",
                              help="Invert black/white image",
                              widget="CheckBox", default=False,
                              action="store_true")
    out_settings.add_argument(
        "--post_filter", metavar="Apply postprocessing filter",
        help="This is to remove small noise and extraneous lines ("
             "only applicable to extremely detailed images)",
        action="store_true", widget="CheckBox")
    resize_settings = parser.add_argument_group("4. Resize settings")
    resize_settings.add_argument(
        "--width", metavar="Width",
        help="Width (inches) (0: no change to current dimensions)",
        action="store", default=0, type=float)
    resize_settings.add_argument(
        "--height", metavar="Height",
        help="Height (inches) (0: no change to current dimensions)",
        action="store", default=0, type=float)
    resize_settings.add_argument(
        "--resolution", metavar="Pixel Resolution (ppi)",
        help="PPI (Default: 96); Tune output resolution here",
        action="store", default=96, type=int)
    return parser


@Gooey(
    program_name="Outline Extraction",
    program_description=("Pipeline for extracting image outlines "
                         "from photos and other graphics"),
    navigation="tabbed"
)
def main():
    gui_manager = GUI(parser().parse_args())
    gui_manager.run()
    return None


if __name__ == "__main__":
    main()
