# Local output structure

These files are generated locally and excluded from GitHub.

```text
models/
└── knn/
    ├── knn_model.pkl
    ├── label_encoder.pkl
    └── label_map.npy

artifacts/
├── processed/
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   └── y_test.npy
├── diagnostics/
│   └── generated_check_images.png
└── metrics/
    └── model_metrics.npz
```

Commit source code and documentation, not generated model files, arrays, or visual-check images.
