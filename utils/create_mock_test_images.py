from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def create_mock_sad_face(output_dir: str | Path = "test_images") -> Path:
    """Create test_images/sad_face.jpg for quick pipeline smoke tests."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / "sad_face.jpg"

    image = np.full((480, 640, 3), (210, 220, 235), dtype=np.uint8)

    face_center = (320, 235)
    cv2.circle(image, face_center, 135, (185, 200, 225), thickness=-1)
    cv2.circle(image, (270, 205), 18, (35, 45, 60), thickness=-1)
    cv2.circle(image, (370, 205), 18, (35, 45, 60), thickness=-1)
    cv2.ellipse(image, (320, 305), (55, 32), 0, 200, 340, (35, 45, 60), thickness=6)
    cv2.line(image, (245, 170), (290, 185), (35, 45, 60), thickness=5)
    cv2.line(image, (395, 170), (350, 185), (35, 45, 60), thickness=5)
    cv2.putText(
        image,
        "mock sad face",
        (205, 430),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (70, 80, 100),
        2,
        cv2.LINE_AA,
    )

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"mock image write failed: {output_path}")
    return output_path


def main() -> None:
    output_path = create_mock_sad_face()
    print(f"Created mock test image: {output_path}")


if __name__ == "__main__":
    main()