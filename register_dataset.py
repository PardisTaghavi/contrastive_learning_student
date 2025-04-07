import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.cityscapes import load_cityscapes_instances

# Define standard Cityscapes colors (manually added)
CITYSCAPES_THING_CLASSES = [
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

CITYSCAPES_THING_COLORS = [
    [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
]

def register_cityscapes_subset():
    cityscapes_root = "/media/avalocal/Samsung_T5/cityscapes"  # Root directory
    image_dir = os.path.join(cityscapes_root, "leftImg8bit/train")  # Image folder
    gt_dir = os.path.join(cityscapes_root, "gtFine/train")  # Ground truth folder

    dataset_name = "cityscapes_KD_train"

    # Register dataset
    DatasetCatalog.register(
        dataset_name, 
        lambda: load_cityscapes_instances(image_dir, gt_dir, from_json=False, to_polygons=True)
    )

    # Set metadata
    MetadataCatalog.get(dataset_name).set(
        thing_classes=CITYSCAPES_THING_CLASSES,  # Cityscapes objects
        thing_colors=CITYSCAPES_THING_COLORS,  # Manually added colors
        evaluator_type="cityscapes_instance_seg",
        ignore_label=255,
        dirname=cityscapes_root,
        json_file="",  # Not using JSON annotations
        image_root=image_dir,
    )

if __name__ == "__main__":
    register_cityscapes_subset()
    print(f"Dataset '{'cityscapes_KD_train'}' registered successfully!")

