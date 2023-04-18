import os
from .register_dm8_seg import dm_train_dicts, VOC_CLASSES, VOC_PALETTE
from detectron2.data import DatasetCatalog, MetadataCatalog

def register_dm8_2_seg(root):
    train_name = 'dm8_2_seg_train'
    

    DatasetCatalog.register(train_name, lambda root=root, split='imagesets/2/good.txt': dm_train_dicts(root))
    MetadataCatalog.get(train_name).set(
        stuff_classes=VOC_CLASSES,
        stuff_colors=VOC_PALETTE,
        ignore_label=255,
    )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_dm8_2_seg(_root)