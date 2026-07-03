# Final Project Summary: Physiotherapy Exercise Recognition

## 1. Final Folder Tree
```text
labeling-cv/
├── dataset_raw/                          (Lower-limb raw videos)
├── dataset_raw_upper/                    (Upper-limb raw videos)
├── processed_dataset/                    (Lower-limb tensors & CSVs)
├── processed_dataset_upper/              (Upper-limb tensors & CSVs)
├── dataset/                              (PyTorch DataLoaders)
├── graph/                                (Graph Definitions)
├── preprocessing/                        (Extraction & Splitting Scripts)
├── training/                             (Training Scripts)
├── inference/                            (Inference Scripts)
├── models/                               (Saved Model Checkpoints)
├── results/                              (Lower-limb Evaluation Artifacts)
├── results_upper/                        (Upper-limb Evaluation Artifacts)
└── docs/                                 (Project Documentation)
```

## 2. Dataset Statistics

| Metric | Lower-Limb Pipeline | Upper-Limb Pipeline | Total Combined |
|---|---|---|---|
| **Total Videos** | 65 | 34 | 99 |
| **Number of Classes** | 9 | 4 | 13 |
| **Training Samples** | 52 | 27 | 79 |
| **Test Samples** | 13 | 7 | 20 |

## 3. Model Performance

| Metric | Lower-Limb | Upper-Limb |
|---|---|---|
| **Architecture** | CTR-GCN (V=10) | CTR-GCN (V=8) |
| **Total Parameters** | ~2.16M | ~2.16M |
| **Best Validation Accuracy** | 53.85% | 85.71% |

## 4. Example Inference Commands

**Lower-Limb Inference:**
```bash
python inference/predict_video.py --video path/to/lower_limb_video.mp4
```

**Upper-Limb Inference:**
```bash
python inference/predict_upper_video.py --video path/to/upper_limb_video.mp4
```

## 5. Future Work
1. **Increase Dataset Size:** Expand the training dataset with clinical exercise videos to improve generalization and mitigate overfitting, as the current model is highly expressive but training data is scarce.
2. **Handle Close-up Shots:** Integrate MediaPipe Hands or a targeted arm/hand landmark model for close-up upper limb videos (e.g., elbow stretching) to improve pose detection rates.
3. **Class Expansion:** Add missing upper-limb classes like `bicep_curl` and `tricep_extension`.
4. **Form Correction & Analysis:** Extend the model to analyze range of motion (ROM) and provide form-correction feedback rather than just classifying the exercise.
5. **Real-Time Deployment:** Deploy a live webcam interface for patients doing physiotherapy exercises at home.
