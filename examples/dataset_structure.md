# Local dataset structure

Create this structure locally. Do not upload the image files to a public repository.

```text
data/
├── raw/
│   ├── Rahul/
│   │   ├── image_001.jpg
│   │   └── image_002.jpg
│   ├── Harshit/
│   ├── Sohail/
│   └── Jagadeesh/
├── legacy_augmented/
│   └── <person>/
└── leakage_free/
    ├── original_splits/
    │   ├── train/<person>/
    │   ├── validation/<person>/
    │   └── test/<person>/
    └── training_augmented/<person>/
```

Use only consented photographs. Split original images first, then augment only `training_augmented`.
