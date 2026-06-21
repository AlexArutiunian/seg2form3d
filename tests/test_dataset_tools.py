import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "train" / "prepare_coco.py"
SPEC = importlib.util.spec_from_file_location("prepare_coco", MODULE_PATH)
prepare_coco = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_coco)


def test_center_selection_prefers_closest_mask():
    mask = np.zeros((100, 100), np.uint16)
    mask[45:55, 45:55] = 1
    mask[5:35, 5:35] = 2
    assert prepare_coco.selected_ids(mask, "center") == [1]


def test_center_selection_prefers_larger_when_both_are_near():
    mask = np.zeros((100, 100), np.uint16)
    mask[46:50, 46:50] = 1
    mask[48:60, 48:60] = 2
    assert prepare_coco.selected_ids(mask, "center") == [2]


def test_coco_rle_starts_with_background_count():
    mask = np.zeros((2, 2), bool)
    mask[0, 0] = True
    assert prepare_coco.rle_counts(mask)[0] == 0
