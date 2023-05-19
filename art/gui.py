# GUI (equivalent to argparse) using GooeyParser

import os
from typing import List, Tuple

import numpy as np
from gooey import Gooey, GooeyParser
from matplotlib.pyplot import imsave
from skimage.io import imread

from art.boundary import BoundaryImage
from art.utils import init_logger


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
        self.logger = init_logger("OutlineExtractor")
        self.args = args
        self.parse_args()
        self.load_image()
        self.boundary_handlers = [
            BoundaryImage(rgb_image,
                          photo_correction=self.is_photo,
                          photo_corr_method=self.exposure_algo)
            for rgb_image in self.rgb_images
        ]

    def load_image(self):
        r"""Load RGB(A) image"""
        self.rgb_images: List[np.ndarray] = [
            imread(fname) for fname in self.files]

    def parse_args(self) -> None:
        r"""Parse and save `args` contents"""
        # defaults
        self.files: List[str] = self.args.filename
        # self.filename: str = self.args.filename
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
        self.resolution: int = self.args.resolution if self.args.resolution > 50 else 96
        self.resize_settings: Tuple[int, int] = (
            self.args.width * self.resolution,
            self.args.height * self.resolution)
        if self.args.width <= 0 or self.args.height <= 0:
            self.resize_settings = None

    def run(self):
        for k, boundary_handler in enumerate(self.boundary_handlers):
            self.logger.info(
                f"Starting {os.path.basename(self.files[k])} ("
                f"{k + 1} / {len(self.boundary_handlers)})")
            boundary_image = boundary_handler.run_pipeline(
                n_classes=self.n_classes,
                apply_post_filter=self.apply_post,
                resize_shape=self.resize_settings,
                invert_colors=self.invert,
                transparent_background=self.transp_bg)
            fname = os.path.basename(self.files[k].rsplit('.', 1)[0])
            output_filename = f"{fname}_boundary_mclass_{self.n_classes}"
            if self.resize_settings is not None:
                output_filename += f"__resized"
            output_loc = os.path.join(
                self.output_dir, f"{output_filename}.{self.output_ftype.lower()}")
            imsave(output_loc, boundary_image,
                   cmap="gray", dpi=self.resolution)
            self.logger.info(
                f"Boundary image saved to {output_loc} with dpi={self.resolution}")
        return None


def _dir_present(x: str):
    if x is None or len(x) == 0:
        raise TypeError(f"This field is required")
    return x


def parser() -> GooeyParser:
    r"""GUI parser with inputs similar to ArgumentParser"""
    parser = GooeyParser()
    step1 = parser.add_argument_group("1. Select File(s) and Output Location")
    step1.add_argument("filename", help="Select image(s) to process", metavar="Files",
                       widget="MultiFileChooser", nargs="+")
    step1.add_argument("--output_dir", metavar="Output Directory",
                       widget="DirChooser",
                       required=True,
                       type=_dir_present)
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
                              widget="CheckBox",
                              action="store_true", default=True)
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
        help="Set output resolution (Default: 96 ppi)",
        action="store", default=96, type=int)
    return parser


@Gooey(
    program_name="Outline Extraction",
    program_description=("Pipeline for extracting image outlines "
                         "from photos and other graphics"),
    default_size=(700, 600)
)
def gooey_gui():
    gui_manager = GUI(parser().parse_args())
    gui_manager.run()
    return None
