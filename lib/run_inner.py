import os, sys
import json
import traceback
import logging, argparse

from joblib import Parallel, delayed
from datetime import datetime, timezone, timedelta
from logging.handlers import (
    TimedRotatingFileHandler as logging_TimedRotatingFileHandler,
)
from copy import deepcopy
from utils.utils import str2bool
from shutil import copyfile
from tqdm import tqdm

from ocr.model import Model as TextRecognize


level_level = logging.INFO
logging.basicConfig(
    format="%(filename)s [LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s",
    level=logging.INFO,
)
logging = logging.getLogger("ocr")
logging.setLevel(level_level)


def parse_argument():
    """python run.py -i test_data/ToRecognize -o test_data/Recognized -j 4"""
    parser = argparse.ArgumentParser(description="""Recognize all images in folder""")
    parser.add_argument(
        "-i", "--in_dir", type=str, default="/ToRecognize", help="directories of images"
    )
    parser.add_argument(
        "-o", "--out_dir", type=str, default="/Recognized", help="output directory"
    )
    parser.add_argument(
        "-j",
        "--n_jobs",
        type=int,
        default=1,
        help="the maximum number of concurrently running jobs",
    )
    parser.add_argument(
        "-lidx", "--last_idx", type=int, default=None, help="last idx. don't use it!"
    )
    parser.add_argument(
        "-sb",
        "--shared_bottom",
        type=str2bool,
        nargs="?",
        const="True",
        default=True,
        help="use shared_bottom or not. default True (use)",
    )
    parser.add_argument(
        "-ip",
        "--image_preprocessing",
        type=str2bool,
        nargs="?",
        const="True",
        default=False,
        help="image preprocessing flag",
    )
    args = parser.parse_args()
    return args


def process(config, logger, tqdm_type="default"):
    text_recognizer = TextRecognize(
        N_JOBS=config.get("N_JOBS", 2), logger=logger, tqdm_type=tqdm_type
    )
    return text_recognizer.run(**config)


def main():
    args = parse_argument()
    logging.info("Files analyzing...")
    allowed_extentions = {".png", ".jpg", ".tif"}
    pages = os.listdir(args.in_dir)

    configs = []
    _pages = []
    for page in pages:
        if os.path.splitext(page)[1] not in allowed_extentions:
            logging.warning(
                f"Format of fail ({page}) is incorrect, should be in: {allowed_extentions}"
            )
        else:
            _pages.append(
                (os.path.join(args.in_dir, page), os.path.join(args.out_dir, page))
            )
            config = {
                "page_path": os.path.join(args.in_dir, page),
                "output_dir": os.path.join(args.out_dir, os.path.splitext(page)[0]),
                "N_JOBS": 2,
                "copy_image": True,
                "LAST_IDX": args.last_idx,
                "use_shared_bottom": args.shared_bottom,
                "image_preprocessing_flag": args.image_preprocessing,
                "allowed_extentions": allowed_extentions,
                "collect_images": False,
                "pointsize": 25,
                "page_image_url": None,
            }
            configs.append(config)
    logging.info(f"Pages to process: {_pages}")

    if args.n_jobs == 1:
        for config in tqdm(configs):
            process(config, logging)
    else:
        result = Parallel(n_jobs=args.n_jobs)(
            delayed(process)(config, logging) for config in tqdm(configs)
        )


if __name__ == "__main__":
    main()
