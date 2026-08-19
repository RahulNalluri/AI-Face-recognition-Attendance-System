# Leakage-free Dataset Workflow

1. Place only consented, original photos in `data/raw/<person>/`.
2. Run `python src/build_leakage_free_dataset.py`.
3. The script creates `data/leakage_free/` without changing the original dataset.
4. Use `data/leakage_free/training_augmented/` only for model training.
5. Use `data/leakage_free/original_splits/validation/` for threshold selection and validation.
6. Use `data/leakage_free/original_splits/test/` only once for the final reported result.

Never copy images from validation/test into training, and never augment validation/test images.

For the next collection round, create capture sessions (for example, `day_1`,
`day_2`, `day_3`) and put all images from one session into only one split.
This gives a more realistic test than a random image split.
