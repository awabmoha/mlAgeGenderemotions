from __future__ import annotations

import argparse
import uuid
from dataclasses import dataclass
from pathlib import Path

from recognition_app.config import AppConfig


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run live face recognition from a webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index. Default: 0")
    parser.add_argument("--model-dir", type=Path, default=project_root / "models", help="Directory with model files.")
    parser.add_argument("--data-dir", type=Path, default=project_root / "face_db", help="SQLite DB and face image folder.")
    parser.add_argument("--threshold", type=float, default=0.45, help="Cosine distance threshold. Lower is stricter.")
    parser.add_argument("--embedding-model", default="Facenet", help="DeepFace embedding model name.")
    parser.add_argument("--emotion-interval", type=int, default=30, help="Analyze emotion every N frames per face.")
    parser.add_argument("--hide-landmarks", action="store_true", help="Do not draw dlib facial landmarks.")
    parser.add_argument(
        "--save-unknowns",
        action="store_true",
        help="Automatically save unmatched faces. By default, press s to save the current unknown face.",
    )
    return parser.parse_args()


@dataclass
class PendingUnknownFace:
    image: object
    embedding: object
    age: str
    gender: str
    emotion: str


def main() -> None:
    args = parse_args()

    import cv2

    from recognition_app.database import FaceDatabase, find_best_match
    from recognition_app.vision import RecognitionModels, crop_face

    config = AppConfig(
        project_root=Path(__file__).resolve().parents[1],
        model_dir=args.model_dir.resolve(),
        data_dir=args.data_dir.resolve(),
        camera_index=args.camera,
        match_threshold=args.threshold,
        embedding_model=args.embedding_model,
        emotion_interval=max(1, args.emotion_interval),
        show_landmarks=not args.hide_landmarks,
    )
    config.validate_model_files()
    config.image_dir.mkdir(parents=True, exist_ok=True)

    print("Loading recognition models...")
    models = RecognitionModels(
        age_prototxt=config.age_prototxt,
        age_model=config.age_model,
        gender_prototxt=config.gender_prototxt,
        gender_model=config.gender_model,
        landmark_model=config.landmark_model,
        embedding_model=config.embedding_model,
    )
    database = FaceDatabase(config.db_path)
    records = database.load_records()
    last_unknown_id: str | None = None
    pending_unknown: PendingUnknownFace | None = None
    last_emotion = "Unknown"

    cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {config.camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow("Age Gender Emotion Recognition", cv2.WINDOW_NORMAL)

    print("Controls: q quit | s save current unknown | n rename last saved unknown | l list recent entries")
    frame_number = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break

            frame_number += 1
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = models.detect_faces(gray)

            for face in faces:
                cropped = crop_face(frame, face)
                if cropped is None:
                    continue
                face_roi, (x1, y1, x2, y2) = cropped

                prediction = models.predict_attributes(
                    face_roi,
                    analyze_emotion=frame_number % config.emotion_interval == 0,
                )
                if prediction.emotion != "Unknown":
                    last_emotion = prediction.emotion

                name = "Unknown"
                distance_label = ""
                if prediction.embedding is not None:
                    best_record, best_distance = find_best_match(records, prediction.embedding)
                    if best_record and best_distance is not None and best_distance <= config.match_threshold:
                        name = best_record.name
                        distance_label = f" ({best_distance:.3f})"
                    else:
                        pending_unknown = PendingUnknownFace(
                            image=face_roi.copy(),
                            embedding=prediction.embedding,
                            age=prediction.age,
                            gender=prediction.gender,
                            emotion=last_emotion,
                        )
                        if args.save_unknowns:
                            last_unknown_id = save_unknown_face(database, config, pending_unknown)
                            records = database.load_records()
                            pending_unknown = None

                if config.show_landmarks and models.can_predict_landmarks:
                    try:
                        landmarks = models.predictor(gray, face)
                        for point in landmarks.parts():
                            cv2.circle(frame, (point.x, point.y), 1, (0, 0, 255), -1)
                    except Exception:
                        pass

                label = f"{name}{distance_label} | {prediction.gender} | {prediction.age} | {last_emotion}"
                draw_label(frame, label, x1, y1, x2, y2)

            cv2.putText(
                frame,
                "q quit | s save unknown | n rename saved face | l list db",
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
            )
            cv2.imshow("Age Gender Emotion Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("l"):
                print_recent(database)
            if key == ord("s"):
                if pending_unknown is None:
                    print("No unmatched face is currently available to save.")
                    continue
                last_unknown_id = save_unknown_face(database, config, pending_unknown)
                records = database.load_records()
                pending_unknown = None
            if key == ord("n"):
                if last_unknown_id is None:
                    print("No saved unknown face to rename. Press s while an unknown face is visible first.")
                    continue
                new_name = sanitize_name(input(f"Rename {last_unknown_id} to: "))
                if new_name:
                    database.rename_face(last_unknown_id, new_name)
                    records = database.load_records()
                    print(f"Renamed {last_unknown_id} to {new_name}.")
                    last_unknown_id = None
    finally:
        cap.release()
        database.close()
        cv2.destroyAllWindows()


def save_unknown_face(database: FaceDatabase, config: AppConfig, pending: PendingUnknownFace) -> str:
    import cv2

    face_id = "unknown_" + uuid.uuid4().hex[:8]
    image_path = config.image_dir / f"{face_id}.jpg"
    cv2.imwrite(str(image_path), pending.image)
    database.insert_face(
        face_id=face_id,
        name=face_id,
        embedding=pending.embedding,
        image_path=str(image_path),
        age=pending.age,
        gender=pending.gender,
        emotion=pending.emotion,
    )
    print(f"Saved {face_id}. Press n to rename it.")
    return face_id


def sanitize_name(raw_name: str) -> str:
    return " ".join(raw_name.strip().split())[:60]


def draw_label(frame, label: str, x1: int, y1: int, x2: int, y2: int) -> None:
    import cv2

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 214, 255), 2)
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    top = max(0, y1 - text_h - 12)
    cv2.rectangle(frame, (x1, top), (x1 + text_w + 12, y1), (0, 0, 0), -1)
    cv2.putText(frame, label, (x1 + 6, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def print_recent(database: FaceDatabase) -> None:
    print("Recent DB entries:")
    for face_id, name, image_path, created_at in database.list_recent():
        print(f" - {face_id} | {name} | {Path(image_path).name} | {created_at}")


if __name__ == "__main__":
    main()
