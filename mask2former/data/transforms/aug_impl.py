import random

from detectron2.data import transforms as T
from fvcore.transforms import TransformList, Transform

from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils

from .transform import CopyPasteTransform

import numpy as np
from PIL import Image

from typing import List

import os.path as osp


class CopyPaste(T.Augmentation):
    """
    Scale jittor and Copy paste.
    Refer to https://arxiv.org/abs/2012.07177.
    """

    def __init__(self,
                 dataset_name,
                 blend=True,
                 sigma=3,
                 resize_range=(0.8, 1.25),
                 crop_size=(0.5, 0.5),
                 pad_value=128,
                 seg_pad_value=255,  # ignore label
                 flip_prob=0.5,
                 interp=Image.BILINEAR,
                 prob=0.5,
                 pre_augs=[],
                 use_scale_jittor=False
                 ):
        _meta = MetadataCatalog.get(dataset_name)
        self.file_list = _meta.file_list
        self.img_dir = _meta.img_dir
        self.seg_dir = _meta.seg_dir
        self.img_suffix = _meta.img_suffix
        self.seg_suffix = _meta.seg_suffix
        self._init(locals())

    def _get_paste_img_seg(self):
        name = random.choice(self.file_list)
        img_path = osp.join(self.img_dir, name + self.img_suffix)
        seg_path = osp.join(self.seg_dir, name + self.seg_suffix)
        paste_img = utils.read_image(img_path, "RGB")
        paste_seg = utils.read_image(seg_path)

        # apply pre augs
        if self.pre_augs:
            aug_input = T.AugInput(paste_img, sem_seg=paste_seg)
            aug_input, _ = T.apply_transform_gens(self.pre_augs, aug_input)
            paste_img, paste_seg = aug_input.image, aug_input.sem_seg

        # apply scale jittor
        for transform in self._get_scale_jittor(paste_img.shape[:2]):
            paste_img = transform.apply_image(paste_img)
            paste_seg = transform.apply_segmentation(paste_seg)
        return paste_img, paste_seg

    def _align_shape(self, paste_img, paste_seg, image_shape):
        paste_shape = paste_img.shape[:2]
        max_h, max_w = max(paste_shape[0], image_shape[0]), max(paste_shape[1], image_shape[1])

        img_padh, img_padw = max_h - image_shape[0], max_w - image_shape[1]
        if img_padh == 0 and img_padw == 0:
            img_transform = T.NoOpTransform()
        else:
            img_transform = T.PadTransform(0, 0, img_padw, img_padh, image_shape[1], image_shape[0], self.pad_value,
                                           self.seg_pad_value)

        paste_padh, paste_padw = max_h - paste_shape[0], max_w - paste_shape[1]
        if paste_padh > 0 or paste_padw > 0:
            paste_transform = T.PadTransform(0, 0, paste_padw, paste_padh, paste_shape[1], paste_shape[0],
                                             self.pad_value, self.seg_pad_value)
            paste_img = paste_transform.apply_image(paste_img)
            paste_seg = paste_transform.apply_segmentation(paste_seg)

        return paste_img, paste_seg, img_transform

    def _get_copy_paste(self, paste_img, paste_seg):
        if self._rand_range() < self.prob:
            return T.NoOpTransform()
        cls_id = np.unique(paste_seg)
        fore_cls_id = cls_id[np.logical_and(cls_id > 0, cls_id != self.seg_pad_value)]
        if len(fore_cls_id) == 0:
            return T.NoOpTransform()
        choosed_id = random.choice(fore_cls_id)
        alpha = paste_seg == choosed_id
        return CopyPasteTransform(
            alpha,
            paste_img,
            paste_seg,
            self.blend,
            self.sigma
        )

    def _get_scale_jittor(self, image_shape) -> List[Transform]:
        if not self.use_scale_jittor:
            return [T.NoOpTransform()]
        transforms = []
        # Resize
        h, w = image_shape
        resize_ratio = np.random.uniform(*self.resize_range)
        new_h, new_w = h * resize_ratio, w * resize_ratio
        transforms.append(T.ResizeTransform(h, w, new_h, new_w, self.interp))
        # Crop
        crop_size = np.asarray(self.crop_size, dtype=np.float32)
        ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
        croph, cropw = int(new_h * ch + 0.5), int(new_w * cw + 0.5)
        croph, cropw = min(croph, h), min(cropw, w)
        assert new_h >= croph and new_w >= cropw, "Shape computation in {} has bugs.".format(self)
        h0 = np.random.randint(new_h - croph + 1)
        w0 = np.random.randint(new_w - cropw + 1)
        transforms.append(T.CropTransform(w0, h0, cropw, croph))
        if self._rand_range() < self.flip_prob:
            transforms.append(T.HFlipTransform(w))
        return transforms

    def get_transform(self, image: np.ndarray, sem_seg: np.ndarray) -> TransformList:
        transforms = []
        image_shape = image.shape[:2]
        transforms.extend(self._get_scale_jittor(image_shape))
        paste_img, paste_seg = self._get_paste_img_seg()
        paste_img, paste_seg, tf = self._align_shape(paste_img, paste_seg, image_shape)
        transforms.append(tf)
        cp = self._get_copy_paste(paste_img, paste_seg)
        transforms.append(cp)
        return TransformList(transforms)
