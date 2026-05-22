# Real-Time Face Analytics

Live webcam face detection, age/gender estimation, emotion recognition, and local face matching using OpenCV, dlib, DeepFace, and SQLite.

This is an old student project that has been cleaned up into a safer GitHub-ready version. It is useful as a computer-vision portfolio project, not as a production identity system.

## What It Does

- Detects faces from a local webcam feed.
- Predicts age range and gender with OpenCV DNN Caffe models.
- Predicts facial emotion with DeepFace.
- Creates face embeddings and matches them against a local SQLite database.
- Lets you save an unknown face only when you press `s`.
- Lets you rename the last saved unknown face with `n`.
- Keeps saved face images and biometric embeddings local by default.

## Important Limits

Age, gender, emotion, and face matching can be wrong. Lighting, camera quality, pose, expression, model bias, and threshold settings all affect the result.

Do not use this project for hiring, policing, access control, medical decisions, school discipline, or any other high-stakes decision. Always get consent before saving someone else's face data.

## Project Structure

```text
recognition_app/       Maintained Python package
scripts/               Small run wrapper
models/                Model notes plus small prototxt files
face_db/               Local runtime data, ignored by Git
database/              Legacy private data, ignored by Git
snapshots/             Legacy private images, ignored by Git
docs/                  Architecture and publishing notes
```

The old React mock dashboard is not included in the clean GitHub package because it used simulated/random detections. The maintained app is the real OpenCV window version.

## Setup

Use Python 3.10 or 3.11 if possible. Some ML packages may not support the newest Python versions immediately.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The core install runs OpenCV face detection plus age/gender estimation. For emotion, identity embeddings, and dlib landmarks, install the full optional set:

```powershell
pip install -r requirements-full.txt
```

If `dlib` is hard to install on Windows, the app can still run without it using OpenCV's built-in Haar face detector. Install CMake/Visual Studio Build Tools or use Conda only if you want the dlib landmark path.

## Model Files

Place these files in `models/`:

- `age_deploy.prototxt`
- `age_net.caffemodel`
- `gender_deploy.prototxt`
- `gender_net.caffemodel`

Optional for dlib landmarks:

- `shape_predictor_68_face_landmarks.dat`

Large model binaries are intentionally ignored by Git. See [models/README.md](models/README.md).

## Run

```powershell
python -m recognition_app.app
```

Useful options:

```powershell
python -m recognition_app.app --camera 1
python -m recognition_app.app --threshold 0.40
python -m recognition_app.app --hide-landmarks
python -m recognition_app.app --model-dir "C:\path\to\models"
```

Controls while the camera window is open:

- `q`: quit
- `s`: save the current unmatched face
- `n`: rename the last saved unknown face
- `l`: list recent saved entries in the terminal

Automatic saving is off by default for privacy. If you really want the old behavior, run with `--save-unknowns`.

## Setup Check

Run this any time the app does not start:

```powershell
python scripts\check_setup.py
```

It reports missing Python packages and missing model files without trying to open the webcam.

## GitHub Publishing

Do not upload this working folder directly because it may contain private runtime data. Generate a clean sibling folder instead:

```powershell
.\scripts\prepare_github_package.ps1
```

Then publish the generated `mlAgeGenderemotions-github-ready` folder.

Before publishing, run:

```powershell
python scripts\check_setup.py
python -m compileall recognition_app
.\scripts\prepare_github_package.ps1
```

See [docs/GITHUB_CHECKLIST.md](docs/GITHUB_CHECKLIST.md) for the final safety checklist.

## License

MIT License. See [LICENSE](LICENSE).
