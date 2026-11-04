import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path


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

def draw_yolo_boxes(
    ax,
    image,
    label_path: Path,
    class_names: dict,
    color: str = "red",
):
    """
    Draw YOLO bounding boxes on an image.

    Args:
        ax: Matplotlib axis.
        image: PIL image.
        label_path: Path to YOLO .txt annotation.
        class_names: Mapping of class IDs to class names.
        color: Bounding box color.
    """

    image_width, image_height = image.size

    ax.imshow(image)

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:

        values = line.strip().split()

        if len(values) != 5:
            continue

        class_id, xc, yc, w, h = map(float, values)

        # Convert normalized coordinates to pixels
        x_center = xc * image_width
        y_center = yc * image_height

        box_width = w * image_width
        box_height = h * image_height

        x1 = x_center - box_width / 2
        y1 = y_center - box_height / 2

        rect = patches.Rectangle(
            (x1, y1),
            box_width,
            box_height,
            linewidth=1,
            edgecolor=color,
            facecolor="none",
        )

        ax.add_patch(rect)

        category = class_names[int(class_id)]

        ax.text(
            x1,
            y1,
            category,
            color="white",
            fontsize=7,
            bbox={
                "facecolor": color,
                "alpha": 0.7,
                "pad": 2,
            },
        )

    ax.axis("off")
    
import json


def visualize_comparison(
    annotation_path: Path,
    split: str,
    source_dir=Path("/home/etman/etman/VisRAG-Papers/data"),
    yolo_dir=Path("/home/etman/etman/VisRAG-Papers/yolo_format_dataset"),
    save_dir: Path = Path("plots"),
    figsize: tuple = (20, 14),
    show_labels: bool = True,
    dpi: int = 150,
):
    """
    Compare original DocLayNet annotations against converted YOLO annotations.
    Save the comparison plot inside plots/<split>/.
    """

    # Original paths
    original_images_dir = source_dir / split / "images"

    # YOLO paths
    yolo_images_dir = yolo_dir / "images" / split
    yolo_labels_dir = yolo_dir / "labels" / split

    # Load original annotation
    with open(annotation_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    # Locate original image
    image_path = next(
        (
            original_images_dir / f"{annotation_path.stem}{ext}"
            for ext in [".png", ".jpg", ".jpeg"]
            if (original_images_dir / f"{annotation_path.stem}{ext}").exists()
        ),
        None,
    )

    if image_path is None:
        raise FileNotFoundError(
            f"Image not found: {annotation_path.stem}"
        )

    # Load image
    image = Image.open(image_path).convert("RGB")

    # YOLO label
    label_path = yolo_labels_dir / f"{annotation_path.stem}.txt"

    if not label_path.exists():
        raise FileNotFoundError(f"YOLO label not found: {label_path}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # -------------------------
    # Original annotations
    # -------------------------
    axes[0].imshow(image)

    for item in annotation["form"]:
        x, y, w, h = map(float, item["box"])

        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(float(image.width), x + w)
        y2 = min(float(image.height), y + h)

        if x2 <= x1 or y2 <= y1:
            continue

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1,
            edgecolor="red",
            facecolor="none",
        )

        axes[0].add_patch(rect)

        if show_labels:
            axes[0].text(
                x1,
                y1,
                item["category"],
                color="white",
                fontsize=7,
                bbox=dict(
                    facecolor="red",
                    alpha=0.7,
                    pad=2,
                ),
            )

    axes[0].set_title("Original DocLayNet annotations")
    axes[0].axis("off")

    # -------------------------
    # Converted YOLO annotations
    # -------------------------
    draw_yolo_boxes(
        ax=axes[1],
        image=image,
        label_path=label_path,
        class_names=CLASS_NAMES,
        # show_labels=show_labels,
    )

    axes[1].set_title("Converted YOLO annotations")

    plt.tight_layout()

    # -------------------------
    # Save visualization
    # -------------------------
    output_dir = Path(save_dir) / split
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / f"{annotation_path.stem}.png"

    fig.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=dpi,
    )

    plt.close(fig)

    # -------------------------
    # Statistics
    # -------------------------
    with open(label_path, "r", encoding="utf-8") as f:
        yolo_objects = len(f.readlines())

    print(f"Image: {annotation_path.stem}")
    print(f"Original objects: {len(annotation['form'])}")
    print(f"YOLO objects: {yolo_objects}")
    print(f"Saved to: {save_path}")

def visualize_document(
    annotation_path: Path,
    images_dir: Path,
    save_dir: str = "plots",
    show_labels: bool = True,
    figsize: tuple = (14, 18),
):
    """
    Visualize a document page with its block-level bounding boxes
    and save the visualization.
    """

    with open(annotation_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    image_path = images_dir / f"{annotation_path.stem}.png"
    image = Image.open(image_path).convert("RGB")

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)

    for item in annotation["form"]:
        x, y, w, h = item["box"]
        category = item["category"]

        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=1.2,
            edgecolor="red",
            facecolor="none",
        )

        ax.add_patch(rect)

        if show_labels:
            ax.text(
                x,
                y,
                category,
                color="white",
                fontsize=7,
                bbox={
                    "facecolor": "red",
                    "alpha": 0.7,
                    "pad": 2,
                },
            )

    ax.axis("off")
    plt.tight_layout()

    # Create output directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save visualization using the annotation filename
    save_path = save_dir / f"{annotation_path.stem}.png"

    fig.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0,
        dpi=150,
    )

    plt.close(fig)

    print(f"Visualization saved to: {save_path}")