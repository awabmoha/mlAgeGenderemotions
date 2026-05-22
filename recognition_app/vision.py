from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import dlib
except ImportError:
    dlib = None

try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None

from recognition_app.config import AGE_BUCKETS_10, AGE_BUCKETS_8, GENDER_LABELS


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int

    def left(self) -> int:
        return self.x

    def top(self) -> int:
        return self.y

    def right(self) -> int:
        return self.x + self.w

    def bottom(self) -> int:
        return self.y + self.h


@dataclass(frozen=True)
class Prediction:
    age: str
    gender: str
    emotion: str
    embedding: np.ndarray | None


class RecognitionModels:
    def __init__(
        self,
        age_prototxt: Path,
        age_model: Path,
        gender_prototxt: Path,
        gender_model: Path,
        landmark_model: Path,
        embedding_model: str,
    ) -> None:
        self.age_net = cv2.dnn.readNetFromCaffe(str(age_prototxt), str(age_model))
        self.gender_net = cv2.dnn.readNetFromCaffe(str(gender_prototxt), str(gender_model))
        self.detector = None
        self.predictor = None
        self.haar_detector = None
        if dlib is not None:
            self.detector = dlib.get_frontal_face_detector()
            if landmark_model.exists():
                self.predictor = dlib.shape_predictor(str(landmark_model))
        else:
            haar_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self.haar_detector = cv2.CascadeClassifier(str(haar_path))
        self.embedding_model = embedding_model

    @property
    def can_predict_landmarks(self) -> bool:
        return self.predictor is not None

    def detect_faces(self, gray_frame: np.ndarray) -> list[object]:
        if self.detector is not None:
            return list(self.detector(gray_frame))
        if self.haar_detector is None:
            return []
        detections = self.haar_detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )
        return [FaceBox(int(x), int(y), int(w), int(h)) for x, y, w, h in detections]

    def predict_attributes(self, face_bgr: np.ndarray, analyze_emotion: bool = True) -> Prediction:
        age, gender = self._predict_age_gender(face_bgr)
        emotion = self._predict_emotion(face_bgr) if analyze_emotion else "Unknown"
        embedding = self._create_embedding(face_bgr)
        return Prediction(age=age, gender=gender, emotion=emotion, embedding=embedding)

    def _predict_age_gender(self, face_bgr: np.ndarray) -> tuple[str, str]:
        try:
            resized = cv2.resize(face_bgr, (227, 227))
            blob = cv2.dnn.blobFromImage(
                resized,
                1.0,
                (227, 227),
                (78.426, 87.769, 114.896),
                swapRB=False,
            )

            self.gender_net.setInput(blob)
            gender_predictions = self.gender_net.forward()
            gender = GENDER_LABELS[int(np.argmax(gender_predictions[0]))]

            self.age_net.setInput(blob)
            age_predictions = self.age_net.forward()
            age_buckets = AGE_BUCKETS_10 if age_predictions.shape[1] >= 10 else AGE_BUCKETS_8
            age = age_buckets[int(np.argmax(age_predictions[0]))]
            return age, gender
        except Exception as exc:
            print(f"[age/gender] {exc}")
            return "Unknown", "Unknown"

    def _predict_emotion(self, face_bgr: np.ndarray) -> str:
        if DeepFace is None:
            return "Unknown"
        try:
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            result = DeepFace.analyze(
                face_rgb,
                actions=["emotion"],
                enforce_detection=False,
                prog_bar=False,
            )
            item = result[0] if isinstance(result, list) and result else result
            if not isinstance(item, dict):
                return "Unknown"
            return str(item.get("dominant_emotion") or item.get("emotion", {}).get("dominant_emotion") or "Unknown")
        except Exception as exc:
            print(f"[emotion] {exc}")
            return "Unknown"

    def _create_embedding(self, face_bgr: np.ndarray) -> np.ndarray | None:
        if DeepFace is None:
            return None
        try:
            result = DeepFace.represent(
                face_bgr,
                model_name=self.embedding_model,
                enforce_detection=False,
                detector_backend="opencv",
                prog_bar=False,
            )
            if isinstance(result, list) and result and isinstance(result[0], dict):
                return np.array(result[0]["embedding"], dtype=np.float32)
            return np.array(result, dtype=np.float32).flatten()
        except Exception as exc:
            print(f"[embedding] {exc}")
            return None


def crop_face(frame: np.ndarray, face: object) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    height, width = frame.shape[:2]
    x1 = max(0, face.left())
    y1 = max(0, face.top())
    x2 = min(width, face.right())
    y2 = min(height, face.bottom())
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)
