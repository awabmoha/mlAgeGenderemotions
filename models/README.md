# Model Files

This folder keeps the small OpenCV deploy files, but large pretrained model binaries are ignored by Git.

Required files:

- `age_deploy.prototxt`
- `age_net.caffemodel`
- `gender_deploy.prototxt`
- `gender_net.caffemodel`

Optional for dlib landmarks:

- `shape_predictor_68_face_landmarks.dat`

The Caffe age/gender files are commonly distributed with OpenCV face age/gender examples. The dlib landmark file is `shape_predictor_68_face_landmarks.dat` from dlib's model zoo.

Check the license/terms of every model file before redistributing it. For a normal GitHub project, it is safer to document the downloads instead of committing the binaries.
