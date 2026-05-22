from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FaceRecord:
    id: str
    name: str
    embedding: np.ndarray
    image_path: str


class FaceDatabase:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                embedding TEXT NOT NULL,
                image_path TEXT,
                created_at TEXT NOT NULL,
                age TEXT,
                gender TEXT,
                emotion TEXT
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def insert_face(
        self,
        face_id: str,
        name: str,
        embedding: np.ndarray,
        image_path: str,
        age: str,
        gender: str,
        emotion: str,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO faces(id, name, embedding, image_path, created_at, age, gender, emotion)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                face_id,
                name,
                _embedding_to_json(embedding),
                image_path,
                dt.datetime.now(dt.timezone.utc).isoformat(),
                age,
                gender,
                emotion,
            ),
        )
        self.connection.commit()

    def rename_face(self, face_id: str, new_name: str) -> None:
        self.connection.execute("UPDATE faces SET name = ? WHERE id = ?", (new_name, face_id))
        self.connection.commit()

    def list_recent(self, limit: int = 15) -> list[tuple[str, str, str, str]]:
        cursor = self.connection.execute(
            """
            SELECT id, name, COALESCE(image_path, ''), created_at
            FROM faces
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return list(cursor.fetchall())

    def load_records(self) -> list[FaceRecord]:
        cursor = self.connection.execute("SELECT id, name, embedding, COALESCE(image_path, '') FROM faces")
        records: list[FaceRecord] = []
        for face_id, name, raw_embedding, image_path in cursor.fetchall():
            try:
                records.append(
                    FaceRecord(
                        id=face_id,
                        name=name,
                        embedding=_json_to_embedding(raw_embedding),
                        image_path=image_path,
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return records


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = left.flatten()
    right = right.flatten()
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return 1.0
    similarity = np.dot(left, right) / (left_norm * right_norm)
    return 1.0 - float(np.clip(similarity, -1.0, 1.0))


def find_best_match(records: list[FaceRecord], embedding: np.ndarray) -> tuple[FaceRecord | None, float | None]:
    best_record: FaceRecord | None = None
    best_distance = 1.0
    for record in records:
        distance = cosine_distance(embedding, record.embedding)
        if distance < best_distance:
            best_record = record
            best_distance = distance
    return best_record, best_distance if best_record is not None else None


def _embedding_to_json(embedding: np.ndarray) -> str:
    return json.dumps(embedding.astype(float).flatten().tolist())


def _json_to_embedding(raw_embedding: str) -> np.ndarray:
    return np.array(json.loads(raw_embedding), dtype=np.float32)
