# Recognition validation input example

Keep real validation photographs in the ignored `data/` directory, not here.
Use this local structure:

```text
data/recognition_validation/
├── enrolled/                 # Optional; otherwise held-out test originals are used
│   ├── Rahul/
│   │   └── new_condition_01.jpg
│   └── Harshit/
│       └── new_condition_01.jpg
└── unknown/                  # Genuinely unenrolled, consented participants
    ├── unknown_01.jpg
    └── unknown_02.jpg
```

Each image should contain one clear face. Use new capture conditions rather
than copies or augmentations of training photographs. Never commit biometric
images, generated embeddings, or validation reports to Git.
