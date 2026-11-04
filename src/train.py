import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from ultralytics import YOLO

from utils.load_config import load_config


def train(config: dict) -> None:
    """Train a YOLO model using the supplied configuration."""

    dataset_path = Path(config["dataset"]["data_yaml"])

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {dataset_path}"
        )

    model_config = config["model"]
    train_config = config["training"]
    output_config = config["output"]

    # Initialize model
    model = YOLO(model_config["architecture"])

    # Train
    model.train(
        data=str(dataset_path),
        epochs=train_config["epochs"],
        imgsz=train_config["image_size"],
        batch=train_config["batch_size"],
        workers=train_config["workers"],
        # device=train_config["device"],
        patience=train_config["patience"],
        deterministic=train_config["deterministic"],
        plots=True,
        project=output_config["project"],
        name=output_config["experiment_name"],
        exist_ok=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO on DocLayNet."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to experiment configuration.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # args = parse_args()
    config = load_config("/home/etman/etman/VisRAG-Papers/config/config.yaml")
    train(config)