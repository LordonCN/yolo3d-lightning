""" Evaluating Code """

from typing import List
from glob import glob
import numpy as np

import dotenv
import hydra
from omegaconf import DictConfig
import os
import sys
import pyrootutils
import src.utils
from src.utils.utils import KITTIObject
from tqdm import tqdm

import src.utils.kitti_common as kitti
from src.utils.eval import get_official_eval_result

log = src.utils.get_pylogger(__name__)

dotenv.load_dotenv(override=True)

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


def get_evaluation(pred_path: str, gt_path: str, classes: List = [0]):
    """Evaluate results"""
    val_ids = [
        int(res.split("/")[-1].split(".")[0])
        for res in sorted(glob(os.path.join(pred_path, "*.txt")))
    ]
    log.info(" EVALUATE: labels number is ", len(val_ids))
    pred_annos = kitti.get_label_annos(pred_path, val_ids)
    gt_annos = kitti.get_label_annos(gt_path, val_ids)

    # compute mAP
    results = get_official_eval_result(
        gt_annos=gt_annos, dt_annos=pred_annos, current_classes=classes
    )

    return results


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="evaluate.yaml")
def main(config: DictConfig):
    
    get_evaluation(config.get("pred_dir"), config.get("gt_dir"), config.get("classes"))

if __name__ == "__main__":
    
    main()
