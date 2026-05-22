from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REQUIRED_MODULES = {
    "cv2": "opencv-python",
    "numpy": "numpy",
}

OPTIONAL_MODULES = {
    "dlib": "dlib for dlib face detection and landmarks",
    "deepface": "deepface for emotion and identity embeddings",
}

REQUIRED_MODELS = [
    "age_deploy.prototxt",
    "age_net.caffemodel",
    "gender_deploy.prototxt",
    "gender_net.caffemodel",
]

OPTIONAL_MODELS = [
    "shape_predictor_68_face_landmarks.dat",
]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ok = True

    print(f"Python: {sys.version.split()[0]}")
    print(f"Project: {root}")
    print()

    print("Python dependencies:")
    for module_name, package_name in REQUIRED_MODULES.items():
        found = importlib.util.find_spec(module_name) is not None
        status = "ok" if found else f"missing - install {package_name}"
        print(f"  {module_name}: {status}")
        ok = ok and found

    print()
    print("Optional enhancement packages:")
    for module_name, description in OPTIONAL_MODULES.items():
        found = importlib.util.find_spec(module_name) is not None
        status = "ok" if found else f"missing - install {description}"
        print(f"  {module_name}: {status}")

    print()
    print("Model files:")
    model_dir = root / "models"
    for file_name in REQUIRED_MODELS:
        path = model_dir / file_name
        found = path.exists()
        status = "ok" if found else "missing"
        print(f"  {file_name}: {status}")
        ok = ok and found

    print()
    print("Optional model files:")
    for file_name in OPTIONAL_MODELS:
        path = model_dir / file_name
        status = "ok" if path.exists() else "missing - landmarks disabled"
        print(f"  {file_name}: {status}")

    print()
    if ok:
        print("Core setup looks ready. Run: python -m recognition_app.app")
        return 0

    print("Setup is incomplete. Follow README.md, then run this check again.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
