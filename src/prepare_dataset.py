import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
from PIL import Image
from pathlib import Path
from collections import defaultdict
import shutil
import yaml

from utils import load_config

DATASET_DIR = Path("/home/etman/etman/VisRAG-Papers/data")
CLASS_NAMES = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title",
}
CLASS_TO_ID = {
    name: class_id
    for class_id, name in CLASS_NAMES.items()
}
SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def convert_bbox(
    bbox: list[float],
    image_width: int,
    image_height: int
):
    """
    Convert a pixel-coordinate XYWH bounding box into
    normalized YOLO coordinates.

    Args:
        bbox: [x, y, width, height] in pixel coordinates.
        image_width: Original image width.
        image_height: Original image height.

    Returns:
        (x_center, y_center, width, height), normalized to [0, 1],
        or None if the box is invalid.
    """
    x, y, width, height = map(float, bbox)
    
    # Reject boxes with invalid dimensions.
    if width <= 0 or height <= 0:
        return None
    
    # Convert XYWH to XYXY.
    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height

    # Clip coordinates to image boundaries.
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(image_width), x2)
    y2 = min(float(image_height), y2)
    
     # Reject boxes completely outside the image.
    if x2 <= x1 or y2 <= y1:
        return None

    # Calculate clipped dimensions.
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    
    # Convert to YOLO normalized format.
    x_center = (x1 + x2) / 2 / image_width
    y_center = (y1 + y2) / 2 / image_height

    normalized_width = clipped_width / image_width
    normalized_height = clipped_height / image_height

    return (
        x_center,
        y_center,
        normalized_width,
        normalized_height,
    )

def convert_annotation(
    annotation_path: Path,
    image_width: int,
    image_height: int,
    selected_classes=None
):
    """
    Read a DocLayNet JSON annotation and convert its objects
    into YOLO label lines.

    Args:
        annotation_path: Path to the JSON annotation.
        image_width: Width of the corresponding image.
        image_height: Height of the corresponding image.
        selected_classes: Optional set of class names to retain.

    Returns:
        A tuple containing:
            - YOLO annotation lines.
            - Filtering statistics.
    """
    with annotation_path.open("r", encoding="utf-8") as file:
        annotation = json.load(file)
        
    yolo_labels = []

    stats = {
        "total": 0,
        "filtered_class": 0,
        "invalid_bbox": 0,
        "kept": 0,
    }
    
    for item in annotation.get("form", []):
        stats["total"] += 1
        category = item.get("category")
        if category not in CLASS_TO_ID:
            stats["invalid_bbox"] += 1
            continue

        class_id = CLASS_TO_ID[category]
        
        # Convert the bounding box.
        yolo_bbox = convert_bbox(
            bbox=item["box"],
            image_width=image_width,
            image_height=image_height,
        )
        
        if yolo_bbox is None:
            stats["invalid_bbox"] += 1
            continue

        x_center, y_center, width, height = yolo_bbox

        label = (
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )
        
        yolo_labels.append(label)
        stats["kept"] += 1

    return yolo_labels, stats

def prepare_split(
    source_dir: Path,
    output_dir: Path,
    split: str,
    selected_classes=None,
):
    """
    split: [train, val, test]
    Convert one DocLayNet split into YOLO format.

    Expected source structure:
        source_dir/
            train/
                images/
                annotations/
            val/
                images/
                annotations/

    Output structure:
        output_dir/
            images/train/
            labels/train/
    """
    image_dir = source_dir / split / "images"
    annotation_dir = source_dir / split / "annotations"
    
    output_image_dir = output_dir / "images" / split
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir = output_dir / "labels" / split
    output_label_dir.mkdir(parents=True, exist_ok=True)
    

    annotation_files = {
        annot_file.stem: annot_file
        for annot_file in annotation_dir.glob("*.json")
    }
    
    image_files = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    
    stats = defaultdict(int)
    for image_file_path in image_files:
        # get the the annotation file which match image_path
        annotation__file_path = annotation_files.get(image_file_path.stem)
        # print(image_path)
        # print(annot_file)
        # read the image
        with Image.open(image_file_path) as image:
            image_height, image_width = image.size
        # print(height, width) # 1025, 1025
        
        # Convert JSON annotation to YOLO labels.
        labels, annotation_stats = convert_annotation(
            annotation_path=annotation__file_path,
            image_width=image_width,
            image_height=image_height,
            selected_classes=selected_classes,
        )
        
        # Copy image to the YOLO dataset.
        shutil.copy2(
            image_file_path,
            output_image_dir / image_file_path.name,
        )

        # Save YOLO annotation file.
        label_path = output_label_dir / f"{image_file_path.stem}.txt"
        
        with label_path.open("w", encoding="utf-8") as file:
            if labels:
                file.write("\n".join(labels) + "\n")

        stats["images"] += 1

        for key, value in annotation_stats.items():
            stats[key] += value
            
    print(f"\nSplit: {split}")
    print(f"Images converted: {stats['images']}")
    print(f"Annotations processed: {stats['total']}")
    print(f"Annotations kept: {stats['kept']}")
    print(f"Filtered classes: {stats['filtered_class']}")
    print(f"Invalid boxes: {stats['invalid_bbox']}")
    print(f"Images missing annotations: {stats['missing_annotations']}")

    return dict(stats)


def create_data_yaml(output_dir: Path) -> Path:
    """Create the Ultralytics dataset configuration file."""

    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))],
    }

    yaml_path = output_dir / "data.yaml"

    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"\nDataset YAML created: {yaml_path}")

    return yaml_path

def prepare_dataset(config: dict) -> None:
    """Prepare all configured DocLayNet splits."""

    source_dir = Path(config["dataset"]["source_dir"])
    output_dir = Path(config["dataset"]["output_dir"])

    # selected_classes_config = config["dataset"].get("selected_classes")

    # selected_classes = (
    #     set(selected_classes_config)
    #     if selected_classes_config is not None
    #     else None
    # )

    # if selected_classes is not None:
    #     unknown_classes = selected_classes - set(CLASS_NAMES.values())

        # if unknown_classes:
        #     raise ValueError(f"Unknown classes: {unknown_classes}")

    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Source dataset directory not found: {source_dir}"
        )

    all_stats = {}

    for split in config["dataset"]["splits"]:
        all_stats[split] = prepare_split(
            source_dir=source_dir,
            output_dir=output_dir,
            split=split,
        )

    create_data_yaml(output_dir)

    print("\nDataset preparation completed.")

        
    
# if __name__ == "__main__":
#     config = load_config("/home/etman/etman/VisRAG-Papers/config/config.yaml")
#     prepare_dataset(config)