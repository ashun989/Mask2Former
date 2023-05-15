import os
import os.path as osp
import cv2
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
import warnings

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']

VOC_PALETTE = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
               (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
               (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
               (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
               (0, 64, 128)]


def get_image_size(path):
    im = cv2.imread(path)
    return im.shape[:2]


def get_tmp_split_path(dm_root):
    dir = osp.join(dm_root, "imageset")
    os.makedirs(dir, exist_ok=True)
    return osp.join(dm_root, "tmp_all.txt")


def read_file_list(split_path, default_dir, default_suffix, dm_root):
    file_list = []
    if split_path is not None:
        with open(split_path, 'r') as fp:
            while True:
                a_line = fp.readline()
                if not a_line:
                    break
                file_list.append(a_line.strip())
    else:
        tmp_path = get_tmp_split_path(dm_root)
        with open(tmp_path, 'w') as fp:
            for fname in os.listdir(default_dir):
                if fname.endswith(default_suffix):
                    name = osp.splitext(fname)[0]
                    file_list.append(name)
                    fp.write(f"{name}/n")
    return file_list


def dm_train_dicts(dm_root, ann_dir, split):
    # data_info_path = osp.join(dm_root, "data_infos.json")
    # with open(data_info_path, 'r') as fp:
    #     data_info = json.load(fp)

    img_dir = osp.join(dm_root, 'img_dir', 'train')
    seg_dir = osp.join(dm_root, ann_dir)

    file_list = read_file_list(split, seg_dir, '.png', dm_root)

    dataset_dicts = []
    for name in file_list:
        record = {}
        record['image_id'] = int(name)
        filename = f"{record['image_id']:08}.png"
        record['file_name'] = osp.join(img_dir,
                                       filename)
        assert osp.isfile(record['file_name']), f"No such file: {record['file_name']}"
        record['height'] = 512
        record['width'] = 512
        record['sem_seg_file_name'] = osp.join(seg_dir,
                                               filename)
        dataset_dicts.append(record)
    return dataset_dicts


def register_dm_seg(root, train_name, dm_name, ann_dir, split=None):
    if train_name in DatasetCatalog.keys():
        return

    dm_root = osp.join(root, dm_name)
    split_path = get_tmp_split_path(dm_root) if split is None else osp.join(dm_root, split)
    img_dir = osp.join(dm_root, 'img_dir', 'train')
    seg_dir = osp.join(dm_root, ann_dir)

    DatasetCatalog.register(train_name,
                            lambda dm_root=dm_root, ann_dir=ann_dir, split=split_path: dm_train_dicts(
                                dm_root,
                                ann_dir,
                                split))
    MetadataCatalog.get(train_name).set(
        stuff_classes=VOC_CLASSES,
        stuff_colors=VOC_PALETTE,
        ignore_label=255,
        split_path=split_path,
        img_dir=img_dir,
        seg_dir=seg_dir,
        img_suffix='.png',
        seg_suffix='.png'
    )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_dm_seg(_root, "dm10_ann_seg_train", "DiffuseMade10", "out_ann/out_ann_dir/tanh2-0.4-4.0-dcrf-0.05-0.95")
register_dm_seg(_root, "dm10_combine_seg_train", "DiffuseMade10",
                "out_combine/out_ann_dir/tanh2-0.25-4.0-dcrf-0.05-0.95")

register_dm_seg(_root, "dm10_combine_trough-0.2-5000-4.0", "DiffuseMade10",
                "out_combine/out_ann_dir/trough-0.2-5000-4.0-dcrf-0.05-0.95")

register_dm_seg(_root, "dm10_combine_trough2-0.2-5000-4.0", "DiffuseMade10",
                "out_combine/out_ann_dir/trough2-0.2-5000-4.0-dcrf-0.05-0.95")

register_dm_seg(_root, "dm10_combine_trough3-0.2-5000", "DiffuseMade10",
                "out_combine/out_ann_dir/trough3-0.2-5000-dcrf-0.05-0.95")

register_dm_seg(_root, "dm10_cross_valley-3-500", "DiffuseMade10",
                "out_cross/out_ann_dir/valley-3-500-dcrf-0.05-0.95")

register_dm_seg(_root, "dm10_combine_sub-1_seg_train", "DiffuseMade10",
                "out_combine/out_ann_dir/tanh2-0.25-4.0-dcrf-0.05-0.95",
                "imageset/cls_1/good.txt")
register_dm_seg(_root, "dm10_combine_sub-2_seg_train", "DiffuseMade10",
                "out_combine/out_ann_dir/tanh2-0.25-4.0-dcrf-0.05-0.95",
                "imageset/cls_2/good.txt")
register_dm_seg(_root, "dm10_combine_sub-3_seg_train", "DiffuseMade10",
                "out_combine/out_ann_dir/tanh2-0.25-4.0-dcrf-0.05-0.95",
                "imageset/cls_3/good.txt")
register_dm_seg(_root, "dm10_combine_sub-4_seg_train", "DiffuseMade10",
                "out_combine/out_ann_dir/tanh2-0.25-4.0-dcrf-0.05-0.95",
                "imageset/seg_2/good.txt")
register_dm_seg(_root, "dm10_combine_sub-5_seg_train", "DiffuseMade10",
                "out_combine/out_ann_dir/tanh2-0.25-4.0-dcrf-0.05-0.95",
                "imageset/seg_3/good.txt")

register_dm_seg(_root, "dm10_clipes_seg_train", "DiffuseMade10",
                "clipes_pseudo_masks_aug")

register_dm_seg(_root, "dm10_combine_clip_fix_box-nms_p600", "DiffuseMade10",
                "out_combine/out_ann_dir/tanh2-0.25-4.0-dcrf-0.05-0.95",
                "imageset/clip_fix_box-nms_p600/good.txt")

register_dm_seg(_root, "dm10_combine_clip_fix_box-nms_p800", "DiffuseMade10",
                "out_combine/out_ann_dir/tanh2-0.25-4.0-dcrf-0.05-0.95",
                "imageset/clip_fix_box-nms_p800/good.txt")

register_dm_seg(_root, "dm10_combine_clip_fix_box-single_p500", "DiffuseMade10",
                "out_combine/out_ann_dir/tanh2-0.25-4.0-dcrf-0.05-0.95",
                "imageset/clip_fix_box-single_p500/good.txt")

register_dm_seg(_root, "dm10_combine_trough_clip_fix_box-single_p500", "DiffuseMade10",
                "out_combine/out_ann_dir/trough-0.2-5000-4.0-dcrf-0.05-0.95",
                "imageset/clip_fix_box-single_p500/good.txt")

register_dm_seg(_root, "dm10_combine_trough_clip_fix_box-single_p500_self", "DiffuseMade10",
                "out_combine/out_ann_dir/trough-0.2-5000-4.0-dcrf-0.05-0.95",
                "imageset/trough-0.2-5000-4.0_clip_fix_box-single_p500/good.txt")

register_dm_seg(_root, "dm10_combine_trough2_clip_fix_box-single_p500_self", "DiffuseMade10",
                "out_combine/out_ann_dir/trough2-0.2-5000-4.0-dcrf-0.05-0.95",
                "imageset/trough2-0.2-5000-4.0_clip_fix_box-single_p500/good.txt")

register_dm_seg(_root, "dm10_cross_valley_clip_fix_box-single_p500_self", "DiffuseMade10",
                "out_cross/out_ann_dir/valley-3-500-dcrf-0.05-0.95",
                "imageset/croos_valley-3-500_clip_fix_box-single_p500/good.txt")

register_dm_seg(_root, "dm10_combine_blip_fix_box-single_p500", "DiffuseMade10",
                "out_combine/out_ann_dir/tanh2-0.25-4.0-dcrf-0.05-0.95",
                "imageset/blip_fix_box-single_p500/good.txt")

register_dm_seg(_root, "dm10_combine_clip_fix_box-max_p500", "DiffuseMade10",
                "out_combine/out_ann_dir/tanh2-0.25-4.0-dcrf-0.05-0.95",
                "imageset/clip_fix_box-max_p500/good.txt")

# ====================================================================

register_dm_seg(_root, "dm11_ann", "DiffuseMade11",
                "out_ann/out_ann_dir/tanh2-0.25-4.0-dcrf-0.05-0.95")

register_dm_seg(_root, "dm11_combine", "DiffuseMade11",
                "out_combine/out_ann_dir/tanh2-0.15-4.0-dcrf-0.05-0.95")

# ====================================================================

register_dm_seg(_root, "dm12_cross_valley-3-500", "DiffuseMade12",
                "out_cross/out_ann_dir/valley-3-500-dcrf-0.05-0.95")  # 90k

register_dm_seg(_root, "dm12_combine_linear-0.25", "DiffuseMade12",
                "out_combine/out_ann_dir/linear-0.25-dcrf-0.05-0.95")  # 90k

register_dm_seg(_root, "dm12_cross_valley-3-500_clip_fix_box-single_p1500", "DiffuseMade12",
                "out_cross/out_ann_dir/valley-3-500-dcrf-0.05-0.95",
                "imageset/cross_valley-3-500_clip_fix_box-single_p1500/good.txt")  # 60k

register_dm_seg(_root, "dm12_combine_linear-0.25_clip_fix_box-single_p1500", "DiffuseMade12",
                "out_combine/out_ann_dir/linear-0.25-dcrf-0.05-0.95",
                "imageset/combine_linear-0.25_fix_box-single_p1500/good.txt")  # 60k

register_dm_seg(_root, "dm12_cross_valley-3-500_clip_fix_box-single_p4000", "DiffuseMade12",
                "out_cross/out_ann_dir/valley-3-500-dcrf-0.05-0.95",
                "imageset/cross_valley-3-500_clip_fix_box-single_p4000/good.txt")  # 10k
