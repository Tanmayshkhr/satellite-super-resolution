from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def find_images(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def inspect_dataset(folder: str | Path) -> dict[str, object]:
    paths = find_images(folder)
    sizes: Counter[tuple[int, int]] = Counter()
    modes: Counter[str] = Counter()
    random_like = 0
    unreadable = 0
    random_like_checked = 0

    for path in paths:
        try:
            with Image.open(path) as image:
                sizes[image.size] += 1
                modes[image.mode] += 1
                if random_like_checked < 100:
                    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
                    unique_ratio = np.unique(pixels.reshape(-1, 3), axis=0).shape[0] / pixels.shape[0]
                    channel_std = float(pixels.std())
                    if unique_ratio > 0.9 and 70 <= channel_std <= 76:
                        random_like += 1
                    random_like_checked += 1
        except (OSError, ValueError):
            unreadable += 1

    synthetic_names = sum(path.stem.lower().startswith("synth_") for path in paths)
    return {
        "count": len(paths),
        "sizes": dict(sizes),
        "modes": dict(modes),
        "synthetic_names": synthetic_names,
        "random_like": random_like,
        "random_like_checked": random_like_checked,
        "unreadable": unreadable,
    }


def dataset_warning(report: dict[str, object]) -> str | None:
    count = int(report["count"])
    synthetic_names = int(report["synthetic_names"])
    random_like = int(report["random_like"])
    random_like_checked = max(1, int(report["random_like_checked"]))
    if count == 0:
        return "No training images were found. Add real satellite images to data/train/."
    if synthetic_names >= count * 0.8 or random_like >= random_like_checked * 0.8:
        return (
            "The current training set appears to be synthetic/random data "
            f"({count} image(s)); it does not match a satellite-imagery project. "
            "Replace data/train/ with real satellite images before training."
        )
    if synthetic_names >= max(1, count * 0.2):
        return (
            "Many training files appear synthetic. Verify that data/train/ "
            "contains real satellite imagery before training."
        )
    return None
