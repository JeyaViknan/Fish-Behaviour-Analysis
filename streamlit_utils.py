"""Utility helpers for streamlit_app.py.

These functions are intentionally pure/testable and avoid Streamlit imports.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_label_line(line: str) -> Tuple[int, List[float]] | None:
    """Parse one YOLO label line and return (class_id, bbox) or None."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    try:
        class_id = int(parts[0])
        bbox = [float(v) for v in parts[1:5]]
    except (ValueError, TypeError):
        return None

    return class_id, bbox


def compute_class_counts(label_files: Iterable[Path], num_classes: int) -> Tuple[Dict[int, int], int]:
    """Count class instances across label files."""
    class_counts: Dict[int, int] = {i: 0 for i in range(num_classes)}
    total_objects = 0

    for label_path in label_files:
        if not label_path.exists():
            continue
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = parse_label_line(line)
                if parsed is None:
                    continue
                class_id, _ = parsed
                if class_id in class_counts:
                    class_counts[class_id] += 1
                    total_objects += 1

    return class_counts, total_objects


def build_train_val_split(
    image_paths: List[Path],
    train_ratio: float,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """Build deterministic train/val split from image paths."""
    if not image_paths:
        return [], []

    if len(image_paths) == 1:
        return image_paths, image_paths

    rng = random.Random(seed)
    shuffled = image_paths.copy()
    rng.shuffle(shuffled)

    split_idx = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_ratio)))
    train_images = shuffled[:split_idx]
    val_images = shuffled[split_idx:]
    return train_images, val_images


def write_split_files(base_dir: Path, train_images: List[Path], val_images: List[Path]) -> Tuple[Path, Path]:
    """Write train/val image-path txt files and return their paths."""
    splits_dir = base_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_txt = splits_dir / "train.txt"
    val_txt = splits_dir / "val.txt"

    with open(train_txt, "w", encoding="utf-8") as f:
        for path in train_images:
            f.write(f"{path.resolve()}\n")

    with open(val_txt, "w", encoding="utf-8") as f:
        for path in val_images:
            f.write(f"{path.resolve()}\n")

    return train_txt, val_txt


def write_data_yaml(
    target_path: Path,
    base_dir: Path,
    train_ref: Path,
    val_ref: Path,
    class_names: List[str],
) -> None:
    """Write Ultralytics-compatible data yaml."""
    try:
        import yaml  # Optional in test/runtime environments.
    except ModuleNotFoundError:
        yaml = None

    data_config = {
        "path": str(base_dir.resolve()),
        "train": str(train_ref.resolve()),
        "val": str(val_ref.resolve()),
        "test": str(val_ref.resolve()),
        "nc": len(class_names),
        "names": class_names,
    }
    with open(target_path, "w", encoding="utf-8") as f:
        if yaml is not None:
            yaml.safe_dump(data_config, f, sort_keys=False)
        else:
            names = ", ".join(f'"{n}"' for n in class_names)
            f.write(f'path: "{data_config["path"]}"\n')
            f.write(f'train: "{data_config["train"]}"\n')
            f.write(f'val: "{data_config["val"]}"\n')
            f.write(f'test: "{data_config["test"]}"\n')
            f.write(f'nc: {data_config["nc"]}\n')
            f.write(f"names: [{names}]\n")
