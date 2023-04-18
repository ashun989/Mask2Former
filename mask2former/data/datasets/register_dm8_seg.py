import os
import os.path as osp
import cv2
import json
from detectron2.data import DatasetCatalog, MetadataCatalog

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

def dm_train_dicts(root):
    dm_root = osp.join(root, "DiffusionMade8")
    data_info_path = osp.join(dm_root, "data_infos.json")
    with open(data_info_path, 'r') as fp:
        data_info = json.load(fp)

    img_dir = osp.join(dm_root, 'img_dir', 'train')
    seg_dir = osp.join(dm_root, 'out_combine', 'out_ann_dir', 'tanh2-0.25-4.0-dcrf-0.05-0.95')

    dataset_dicts = []
    for di in data_info:
        record = {}
        record['image_id'] = int(di['img_index'])
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

def voc_val_dicts(root):
    voc_root = osp.join(root, 'VOCdevkit', 'VOC2012')
    val_split_path = osp.join(voc_root, 'ImageSets', 'Segmentation', 'val.txt')
    val_names = []
    with open(val_split_path, 'r') as fp:
        while True:
            aline = fp.readline()
            if not aline:
                break
            val_names.append(aline.strip())
    img_dir = osp.join(voc_root, 'JPEGImages')
    seg_dir = osp.join(voc_root, 'SegmentationClassAUG')
    dataset_dicts = []
    for idx, name in enumerate(val_names):
        record = {}
        record['image_id'] = idx
        record['file_name'] = osp.join(img_dir, name + '.jpg')
        assert osp.isfile(record['file_name']), f"No such file: {record['file_name']}"
        record['height'], record['width'] = get_image_size(record['file_name'])
        record['sem_seg_file_name'] = osp.join(seg_dir, name + '.png')
        dataset_dicts.append(record)
    return dataset_dicts

def register_dm8_seg(root):
    train_name = 'dm8_seg_train'
    val_name = 'voc_seg_val'

    DatasetCatalog.register(train_name, lambda root=root: dm_train_dicts(root))
    MetadataCatalog.get(train_name).set(
        stuff_classes=VOC_CLASSES,
        stuff_colors=VOC_PALETTE,
        ignore_label=255,
    )

    DatasetCatalog.register(val_name, lambda root=root: voc_val_dicts(root))
    MetadataCatalog.get(val_name).set(
        stuff_classes=VOC_CLASSES,
        stuff_colors=VOC_PALETTE,
        ignore_label=255,
        evaluator_type="sem_seg",
    )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_dm8_seg(_root)