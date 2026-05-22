# GitHub Safety Checklist

Use this checklist before publishing.

- Generate the clean package with `.\scripts\prepare_github_package.ps1`.
- Publish the generated `mlAgeGenderemotions-github-ready` folder, not the working Google Drive folder.
- Confirm the package does not include `face_db/`, `database/`, `snapshots/`, `.conda/`, `.vs/`, `node_modules/`, or `__pycache__/`.
- Confirm there are no `.db`, `.sqlite`, `.jpg`, `.png`, `.log`, `.caffemodel`, `.dat`, `.h5`, or `.onnx` files unless you intentionally reviewed them.
- Do not commit saved face images or embeddings without explicit consent.
- Do not commit model binaries unless their license clearly allows redistribution.
- Run `python -m compileall recognition_app`.
- Read the README once from the clean package and make sure the setup instructions still match the files included there.

Recommended GitHub description:

`Local webcam face analytics demo using OpenCV, dlib, DeepFace, and SQLite. Includes age/gender/emotion estimation and opt-in local face matching.`
