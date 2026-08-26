# AI-Based Face Recognition Attendance System Using Neural Networks

This project develops an automated attendance system that identifies
registered students from a live camera and records attendance only after
identity confirmation and active liveness verification. It combines face
preprocessing, controlled data augmentation, neural-network recognition,
unknown-person rejection, scheduled attendance checkpoints, and a live local
monitoring dashboard.

## Project objectives

- Automate attendance using facial recognition.
- Reduce proxy and duplicate attendance entries.
- Evaluate recognition under changes in pose, brightness, and appearance.
- Use a leakage-free train/validation/test workflow.
- Compare lightweight baseline models with a convolutional neural network.
- Prepare the system for webcam inference and database integration.

## Current development status

### Phase 1 — Leakage-free dataset preparation

Original photographs are divided into training, validation, and test sets
before augmentation. Only training photographs are augmented. This prevents
transformed copies of validation or test photographs from entering training.

### Phase 2 — CNN baseline

The CNN uses three convolutional blocks with batch normalization, max pooling,
dropout, global average pooling, and a four-class Softmax output. Early stopping,
learning-rate reduction, and best-checkpoint selection are based on validation
loss. Final metrics are calculated on held-out original test photographs.

### Phase 3 — Face-focused few-shot recognition

YuNet detects facial landmarks and aligns each face before recognition. A
pretrained SFace neural network converts the aligned face into a compact
embedding, allowing a new identity to be enrolled from a small local image set.
The local embeddings, labels, downloaded weights, and photographs remain
excluded from Git.

### Unknown rejection, real-time recognition, and liveness

The calibrated cosine threshold rejects insufficiently similar faces as
`Unknown`. Real-time recognition tracks multiple faces and confirms identity
across consecutive frames. A confirmed identity must then complete a randomized
active-liveness sequence containing a blink and a left/right head turn before a
recognition event is emitted.

### Scheduled attendance and live monitoring dashboard

Trusted camera events are sent to a local Flask API and stored in SQLite. A
class can contain multiple attendance checkpoints—for example, a 130-minute
class starting at 9:30 with a 65-minute interval creates checks at 9:30 and
10:35. A student can be marked only once per checkpoint. The login-free local
dashboard displays the camera preview, recognized model identities, checkpoint
counts, and recognition results as they arrive. Camera preview frames remain in
memory and are not written to disk.

## CNN architecture

```text
96 × 96 RGB face
      ↓
Conv2D(32) → BatchNorm → ReLU → Conv2D(32) → MaxPool → Dropout
      ↓
Conv2D(64) → BatchNorm → ReLU → Conv2D(64) → MaxPool → Dropout
      ↓
Conv2D(128) → BatchNorm → ReLU → Conv2D(128) → MaxPool → Dropout
      ↓
Global Average Pooling → Dense(128) → Dropout
      ↓
Softmax identity classification
```

## Repository structure

```text
├── src/                    # Dataset, training, and inference programs
├── scripts/diagnostics/    # Dataset inspection utilities
├── docs/                   # Phase workflows and technical documentation
├── examples/               # Safe example structures for GitHub
├── data/                   # Local biometric dataset; excluded from Git
├── models/                 # Generated trained models; excluded from Git
└── artifacts/              # Metrics and plots; excluded from Git
```

## Installation

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS, activate with `source .venv/bin/activate`.

## Usage

### 1. Prepare the leakage-free dataset

Place consented original images in `data/raw/<identity>/`, then run:

```bash
python src/build_leakage_free_dataset.py
```

### 2. Train and evaluate the CNN

```bash
python src/train_cnn.py --epochs 40 --batch-size 32
```

Training produces a saved Keras model, accuracy/loss history, confusion matrix,
per-class precision, recall, F1-score, and final test accuracy.

### 3. Predict one image

```bash
python src/predict_cnn.py path/to/face.jpg
```

### 4. Train the face-specific few-shot classifier

Download the public OpenCV face models, create aligned crops while preserving
the existing splits, and build the local classifier:

```bash
python src/download_face_models.py
python src/build_face_cropped_dataset.py --overwrite
python src/train_sface_classifier.py
python src/calibrate_unknown_threshold.py
```

Recognize a new image with:

```bash
python src/predict_sface.py path/to/face.jpg
```

Prediction automatically uses the locally calibrated cosine threshold and
returns `Unknown` when the closest enrolled face is not similar enough. The
calibration simulates unknown users by excluding each validation identity from
the comparison. A final deployment threshold still requires testing with
genuinely unenrolled, consented participants.

### 5. Start the local dashboard

Initialize the private local database and start the dashboard:

```bash
python src/manage_attendance.py init
python src/web_app.py
```

Open `http://127.0.0.1:5000`. Manual registration is not required: after a
trained identity passes recognition and liveness for the first time, its exact
SFace label is added to the local database automatically. The application also
synchronizes every label from `models/sface/labels.json` at startup, and every
new attendance session stores its own roster snapshot. The database, local
session secret, and protected camera device token are stored under `instance/`,
which is excluded from Git.

### 6. Create a scheduled class

Use the **Start attendance session** form on the dashboard and set:

- class start and duration;
- checkpoint interval (65 minutes by default);
- checkpoint window (10 minutes by default).

For a class beginning at 9:30, the defaults produce the next checkpoint at
10:35. Recognition outside an open window is audited but does not mark the
student present.

For a quick live test, click **Use 8-minute test preset** before starting the
session. It creates checkpoints at minute 0, 3, and 6, with each checkpoint
open for 2 minutes. The one-minute gaps make it easy to confirm that recognition
is recorded once during each open window and is only audited between windows.

The same repeated-checkpoint workflow can be verified without a webcam and
without modifying the private application database:

```bash
python scripts/diagnostics/verify_repeated_checkpoints.py
```

The verifier uses a temporary isolated database and checks the full sequence:
first mark, duplicate rejection, closed-window rejection, second mark, and
second duplicate rejection.

Open **Session history** to view retained sessions. Each report includes the
saved roster, checkpoint windows, present/absent/open/upcoming status, per-person
and overall percentages, and a checkpoint-wise CSV export. A person becomes
absent only after that checkpoint window closes.

### 7. Manage completed attendance

Open a session report and use **Search roster** to find a person by name, model
label, or registration number. A completed checkpoint can be corrected to
present or absent by opening **Correct**, selecting the effective status, and
entering a reason. Choose **Restore automatic** to return to the original
recognition result.

Corrections never overwrite face-recognition evidence. The original timestamp,
similarity, and liveness-backed attendance row remain stored, while the effective
status is maintained separately. Every correction and restoration records the
previous status, new status, reason, time, and operator in the session's audit
trail. Reports, percentages, checkpoint totals, and CSV exports use the effective
status; manually corrected CSV cells are labelled `(manual)`.

### 8. Start recognition from the dashboard

Leave the camera source on **Auto detect** and click **Start Camera**. The
application checks camera indices 0–3 and uses the first device that returns a
real frame. A specific camera can still be selected when multiple devices are
connected. The dashboard launches the
recognition and liveness pipeline, shows its preview, and displays any startup
error. Click **Stop Camera**
to close the webcam and recognition process cleanly.

Close Windows Camera, Teams, Zoom, and browser camera tabs before starting;
Windows normally permits only one application to control the webcam. Preview
frames are published on a background thread as a latest-frame stream at up to
10 FPS, so a slow browser cannot pause recognition or accumulate delayed frames.

The camera controller now verifies that the device produces a real frame before
reporting it as ready. Brief dropped frames are retried automatically. If the
camera disconnects or stops returning frames, recognition releases the device,
attempts to reconnect, clears stale face/liveness tracking state, and resumes
without restarting the attendance session. The dashboard shows the selected
source, preview health, and recovery count, and hides previews older than four
seconds instead of displaying a frozen image.

For command-line debugging, the recognizer can still be started directly:

```bash
python src/realtime_recognition.py
```

Use another camera with `--source 1`, or test a recorded video with
`--source path/to/video.mp4`. The default pipeline processes every second frame,
confirms an identity across three observations, and applies a ten-second event
cooldown. These controls can be adjusted:

```bash
python src/realtime_recognition.py --process-every 2 --confirmation-frames 3 --cooldown-seconds 10
```

Camera recovery defaults to five tolerated read failures followed by five
reconnection attempts at one-second intervals. These settings can be adjusted:

```bash
python src/realtime_recognition.py --camera-read-failures 5 --camera-reconnect-attempts 5 --camera-reconnect-delay 1
```

For a safe camera-only check that neither records attendance nor writes event
logs, run:

```bash
python src/realtime_recognition.py --headless --max-frames 30 --no-attendance-api --no-event-log --no-ui-frame
```

The dashboard-managed recognizer reads the private device token from
`instance/device_token.txt` and sends passed events to the attendance API. It
retries the database sync every 30 seconds while a
live, confirmed student remains visible, allowing a later hourly checkpoint to
be marked without restarting the camera. Change this with
`--attendance-sync-seconds`, or use `--no-attendance-api` for recognition-only
operation.

Confirmed events are also stored locally in
`artifacts/realtime/recognition_events.jsonl`. Camera frames are displayed but
are not saved. A compressed preview is passed to the local dashboard in memory;
disable it with `--no-ui-frame`. Event logging can be disabled with
`--no-event-log`, and systems without a desktop display can use `--headless`
while retaining the browser preview.

The live window displays the current randomized instruction. A successful event
has the type `recognition_and_liveness_passed`; recognition alone is never
written as an attendance-ready event. The default liveness timeout is 12 seconds
and can be changed with `--liveness-timeout`.

Active blink/head-turn checks provide basic protection against static printed
or screen-displayed photographs. They are not a complete defence against
sophisticated replay or deepfake attacks; a passive anti-spoofing model and
testing with real presentation attacks are still required before deployment.

The current internal evaluation reached 96.15% validation accuracy and 100%
accuracy on 25 held-out originals. Because the test set is small, these figures
describe only the current dataset and must not be interpreted as universal
real-world accuracy.

## Evaluation policy

- Training uses `data/leakage_free/training_augmented/`.
- Model selection uses held-out originals in the validation split.
- The test split is evaluated only after training.
- Accuracy, macro F1-score, per-class metrics, and the confusion matrix are
  reported together.
- The CNN baseline is evaluated as a closed-set classifier; deployed SFace
  inference separately applies a calibrated unknown-person threshold.

## Planned enhancements

- Passive anti-spoofing in addition to the current active liveness challenge.
- Reusable timetables and named class groups for automatic session creation.
- Attendance percentages, shortage alerts, and downloadable reports.
- PostgreSQL, HTTPS, and deployment hardening for multi-device operation.
- Consent, encryption, biometric-data retention, and deletion controls.

## Privacy

Face photographs are biometric data. Use only images collected with informed
consent. Raw images, augmented datasets, trained models, and generated outputs
are excluded from the public repository through `.gitignore`.
