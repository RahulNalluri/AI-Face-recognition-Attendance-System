"""Download public OpenCV YuNet and SFace ONNX weights for local use."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DESTINATION = ROOT / "models" / "pretrained"
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        200_000,
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        30_000_000,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for filename, (url, minimum_size) in MODELS.items():
        destination = DESTINATION / filename
        if destination.exists() and destination.stat().st_size >= minimum_size and not args.force:
            print(f"Already available: {destination}")
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, temporary)
            if temporary.stat().st_size < minimum_size:
                raise RuntimeError(f"Downloaded file is unexpectedly small: {temporary}")
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
