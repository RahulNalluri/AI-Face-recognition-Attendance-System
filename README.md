# AI-Based Face Recognition Attendance System Using Neural Networks

This project develops an automated attendance system that identifies
registered students from facial images and prepares their identities for
attendance logging. The system investigates face preprocessing, controlled
data augmentation, neural-network classification, model evaluation, and the
future integration of unknown-person rejection, liveness detection, and secure
attendance records.

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

## Evaluation policy

- Training uses `data/leakage_free/training_augmented/`.
- Model selection uses held-out originals in the validation split.
- The test split is evaluated only after training.
- Accuracy, macro F1-score, per-class metrics, and the confusion matrix are
  reported together.
- The current CNN is a closed-set classifier and does not yet provide calibrated
  unknown-person recognition.

## Planned enhancements

- Face alignment and a stronger face detector.
- FaceNet or ArcFace embeddings for scalable enrollment.
- Calibrated unknown-person rejection.
- Liveness and presentation-attack detection.
- Real-time webcam recognition.
- Secure database attendance records and duplicate prevention.
- Authentication, consent, encryption, and biometric-data retention controls.

## Privacy

Face photographs are biometric data. Use only images collected with informed
consent. Raw images, augmented datasets, trained models, and generated outputs
are excluded from the public repository through `.gitignore`.
