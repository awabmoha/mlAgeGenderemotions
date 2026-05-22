# Architecture

The maintained app is a local Python/OpenCV program.

## Runtime Flow

1. Open a webcam with OpenCV.
2. Detect faces with dlib's frontal face detector.
3. Crop each detected face from the frame.
4. Predict age and gender with OpenCV DNN Caffe models.
5. Predict emotion with DeepFace on a configurable interval.
6. Create a DeepFace embedding for matching.
7. Compare the embedding with records stored in local SQLite.
8. Draw the live label and bounding box on the camera frame.

## Data Storage

Saved face records live in `face_db/faces.db`. Saved crops live in `face_db/images/`.

The database contains biometric embeddings and should be treated as private data. The clean GitHub package excludes all runtime data folders.

## Recognition Behavior

Matching uses cosine distance. The default threshold is `0.45`; lower values are stricter and reduce false matches, but may show more known people as unknown.

Unknown faces are not saved automatically unless `--save-unknowns` is passed. The safer default is to press `s` when an unmatched face should be saved.
