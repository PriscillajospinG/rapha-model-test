# Dataset Labeling Strategy

To prepare the physiotherapy exercise videos for machine learning, several labeling approaches were considered. After evaluating their strengths and limitations, a **hybrid labeling strategy** was selected as the most suitable approach.

---

## Labeling Approaches Considered

### 1. Expert (Manual) Annotation

Each video clip is manually inspected and labeled with:

* Exercise name
* Repetition count
* Form assessment score
* Other relevant parameters

#### Advantages

* Produces highly accurate labels.
* Provides reliable ground truth data.
* Essential for creating a validation dataset.

#### Limitations

* Time-consuming and labor-intensive.
* Difficult to scale for larger datasets.
* Requires domain knowledge and human effort.

---

### 2. Weak Supervision

Instead of manually labeling every sample, heuristic rules are used to generate labels automatically.

#### Workflow

```text
Video Clips
      ↓
MediaPipe Pose Extraction
      ↓
Pose Vectors
      ↓
Rule-Based Labeling
      ↓
Weak Labels
```

Examples of heuristics include:

* Joint angle thresholds
* Range of motion patterns
* Movement trajectories
* Temporal characteristics of the exercise

#### Advantages

* Significantly reduces manual effort.
* Easily scalable.
* Enables rapid generation of labels.

#### Limitations

* Label quality depends heavily on the heuristics.
* Difficult to capture all exercise variations.
* Incorrect rules can introduce noisy labels.

---

### 3. Auto-Labeling

A trained classifier can be used to assign labels to previously unseen samples.

Possible models include:

* Random Forest
* XGBoost
* Support Vector Machine (SVM)
* Logistic Regression

#### Workflow

```text
Pose Vectors
      ↓
Initial Labeled Dataset
      ↓
Train Classifier
      ↓
Predict Labels for Remaining Samples
```

#### Advantages

* Fast and scalable.
* Suitable for expanding the dataset.
* Requires minimal human intervention.

#### Limitations

* Depends on the quality of existing labels.
* Errors in the training set may propagate.
* Cannot be used effectively without an initial labeled dataset.

---

# Selected Approach: Hybrid Labeling

A combination of **Expert Annotation**, **Weak Supervision**, and **Auto-Labeling** was adopted to leverage the strengths of each approach.

## Motivation

The hybrid approach provides:

* **Accuracy** through expert annotation.
* **Efficiency** through heuristic-based labeling.
* **Scalability** through automated classification.

---

## Proposed Pipeline

```text
Raw Exercise Videos
         ↓
MediaPipe Pose Extraction
         ↓
Small Expert-Annotated Dataset
         ↓
Weak Supervision Rules
         ↓
Initial Labels
         ↓
Train Classifier
(Random Forest / XGBoost)
         ↓
Auto-Label Remaining Samples
         ↓
Human Verification
         ↓
Final Labeled Dataset
```

---

## Advantages of the Hybrid Approach

### High Accuracy

A manually annotated subset serves as reliable ground truth and helps maintain label quality.

### Reduced Human Effort

Only a fraction of the dataset requires manual inspection.

### Scalability

Large amounts of data can be labeled efficiently.

### Faster Dataset Expansion

New exercise categories can be incorporated with minimal additional annotation.

### Continuous Improvement

As more verified labels become available, the classifier can be retrained to improve performance.

### Industry-Relevant Workflow

This approach closely resembles labeling pipelines used in:

* Healthcare AI
* Human Activity Recognition
* Computer Vision
* Sports Analytics
* Rehabilitation Systems

---

## Conclusion

Considering the trade-off between **accuracy**, **efficiency**, and **scalability**, a **hybrid labeling strategy** was chosen for this project. By combining expert annotation, weak supervision, and automated labeling, the dataset can be expanded efficiently while maintaining high-quality labels suitable for downstream machine learning tasks such as:

* Exercise classification
* Repetition counting
* Form assessment
* Pose analysis
* Rehabilitation monitoring
