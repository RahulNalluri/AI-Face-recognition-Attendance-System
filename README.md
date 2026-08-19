# AI-Based Face Recognition Attendance System

Final-year project for automated attendance using face detection, image preprocessing, and neural-network-based classification.

## Repository layout

- `src/` — dataset preparation, augmentation, and training scripts
- `scripts/diagnostics/` — visual and data-quality checks
- `data/` — local datasets (ignored by Git because they contain biometric images)
- `models/` — trained model files (ignored by Git)
- `artifacts/` — generated arrays, metrics, and diagnostic images (ignored by Git)
- `examples/` — safe folder-layout examples for GitHub; no real face images
- `docs/` — project workflow documentation

## Data privacy

Only use face images collected with informed consent. Do not upload raw face datasets or trained biometric artifacts to a public GitHub repository.

The repository includes example folder structures only. Follow the instructions
in `examples/` to place your local data and generated results when running the project.

## Leakage-free workflow

Run `python src/build_leakage_free_dataset.py` to split original images before augmenting the training set. See `docs/DATASET_V2_WORKFLOW.md`.
