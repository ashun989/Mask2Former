import os
import json
import numpy as np
import cv2

VOC_PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
               [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
               [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


def parse_path(path):
    """
    :param path: A file path
    :return: dir, filename, extension
    """
    paths = os.path.split(path)
    dir = os.path.sep.join(paths[:-1])
    filename, extension = os.path.splitext(paths[-1])
    return dir, filename, extension


def reduce_zero_label(gt_semantic_seg):
    # avoid using underflow conversion
    gt_semantic_seg[gt_semantic_seg == 0] = 255
    gt_semantic_seg = gt_semantic_seg - 1
    gt_semantic_seg[gt_semantic_seg == 254] = 255
    return gt_semantic_seg


def get_show_map(img, label, palette, map_ratio=0.5, ignore_idx=255, reduce_zero=False):
    if reduce_zero:
        label = reduce_zero_label(label)
    is_ignore = label == ignore_idx
    label[is_ignore] = 0
    show_map = palette[label]
    show_map[is_ignore] = np.array([255, 255, 255])
    if img is not None:
        img = img * (1 - map_ratio) + show_map * map_ratio
        img = img.astype(np.uint8)
        return img
    return show_map


class OptTargetRecord:
    def __init__(self, opt_target_dir):
        self.opt_target_dir = opt_target_dir

    def _get_info_path(self, name):
        return os.path.join(self.opt_target_dir, f"info_{name}.json")

    def _get_seg_path(self, name):
        return os.path.join(self.opt_target_dir, f"{name}.png")

    def update_info(self, name, update_dict):
        path = self._get_info_path(name)
        info = {}
        if os.path.isfile(path):
            with open(path, 'r') as fp:
                info = json.load(fp)
        info.update(update_dict)
        with open(path, 'w') as fp:
            json.dump(info, fp)

    def read_info(self, name):
        path = self._get_info_path(name)
        info = {}
        if os.path.isfile(path):
            with open(path, 'r') as fp:
                info = json.load(fp)
        return info

    def write_info(self, name, info):
        path = self._get_info_path(name)
        with open(path, 'w') as fp:
            json.dump(info, fp)

    def write_seg(self, name, sem_seg: np.ndarray):
        path = self._get_seg_path(name)
        cv2.imwrite(path, sem_seg)

    def read_seg(self, name):
        path = self._get_seg_path(name)
        if os.path.isfile(path):
            return cv2.imread(path, 0)
        return None
