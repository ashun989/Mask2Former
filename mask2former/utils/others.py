import os
import json
import numpy as np
import cv2


def parse_path(path):
    """
    :param path: A file path
    :return: dir, filename, extension
    """
    paths = os.path.split(path)
    dir = os.path.sep.join(paths[:-1])
    filename, extension = os.path.splitext(paths[-1])
    return dir, filename, extension


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
