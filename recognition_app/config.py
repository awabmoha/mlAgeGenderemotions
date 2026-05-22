from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


AGE_BUCKETS_8 = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60+)",
]

AGE_BUCKETS_10 = [
    "(0-4)",
    "(6-10)",
    "(15-18)",
    "(20-24)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-80)",
    "(85-95)",
    "(100+)",
]

GENDER_LABELS = ["Male", "Female"]


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    model_dir: Path
    data_dir: Path
    camera_index: int = 0
    match_threshold: float = 0.45
    embedding_model: str = "Facenet"
    emotion_interval: int = 30
    show_landmarks: bool = True

    @property
    def db_path(self) -> Path:
        return self.data_dir / "faces.db"

    @property
    def image_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def age_prototxt(self) -> Path:
        return self.model_dir / "age_deploy.prototxt"

    @property
    def age_model(self) -> Path:
        return self.model_dir / "age_net.caffemodel"

    @property
    def gender_prototxt(self) -> Path:
        return self.model_dir / "gender_deploy.prototxt"

    @property
    def gender_model(self) -> Path:
        return self.model_dir / "gender_net.caffemodel"

    @property
    def landmark_model(self) -> Path:
        return self.model_dir / "shape_predictor_68_face_landmarks.dat"

    def validate_model_files(self) -> None:
        required = [
            self.age_prototxt,
            self.age_model,
            self.gender_prototxt,
            self.gender_model,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing model files:\n"
                + "\n".join(f"  - {path}" for path in missing)
                + "\n\nPlace the pretrained files in models/ or pass --model-dir."
            )
