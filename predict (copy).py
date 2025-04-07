import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.cityscapes import load_cityscapes_instances

def register_cityscapes_subset():
    # Change these paths to your dataset directory
    cityscapes_root = "/media/avalocal/Samsung_T5/cityscapes_KD"  # Root folder
    image_dir = os.path.join(cityscapes_root, "leftImg8bit/train")  # Image folder
    gt_dir = os.path.join(cityscapes_root, "gtFine/train")  # Ground truth folder

    dataset_name = "cityscapes_KD_train"  # Name for the dataset
    
    # Register the dataset
    DatasetCatalog.register(
        dataset_name, 
        lambda: load_cityscapes_instances(image_dir, gt_dir, from_json=False, to_polygons=True)
    )

    # Get default Cityscapes metadata and set it for our subset
    cityscapes_metadata = MetadataCatalog.get("cityscapes_fine_instance_seg_train")  # Original metadata
    MetadataCatalog.get(dataset_name).set(
        thing_classes=cityscapes_metadata.thing_classes,  # Use default Cityscapes categories
        thing_colors=cityscapes_metadata.thing_colors,  # Colors for visualization
        evaluator_type="cityscapes_instance_seg",
        ignore_label=255,
        dirname=cityscapes_root,
        json_file="",  # Not using JSON annotations
        image_root=image_dir,
    )

if __name__ == "__main__":
    register_cityscapes_subset()
    print(f"Dataset '{'cityscapes_subset_train'}' registered successfully!")

