notes

1. ⁠1. MediaPipe Holistic is a pre-trained landmark extraction system.
2. CTR-GCN is a Skeleton-Based Action Recognition Model



Converting Physiotherapy Videos into CTR-GCN Training Data

Video → MediaPipe Holistic → Landmark Vectors → CTR-GCN → Model Weights (.pth)

The goal is to convert exercise videos into skeleton-based landmark vectors and use them to train a CTR-GCN model for lower-limb movement analysis.


so -  1st
Suppose we have a squat video:

* File: squat.mp4
* Frame Rate: 30 FPS
* Duration: 3 seconds

Total frames:
30 × 3 = 90 frames
So the model will process 90 individual frames.

Landmark Extraction - 2nd step
Each frame is passed through MediaPipe Holistic.
MediaPipe detects body landmarks and returns coordinates for every joint.

For the lower-limb physiotherapy module, only the required joints are retained.




Pseudo Workflow:

1. Read video frame by frame.
2. Run MediaPipe on each frame.
3. Extract lower-limb joints.
4. Store the joints in a sequence.
5. Convert the sequence into a NumPy tensor.

Output:

tensor.shape = (90,10,3)



Important Research Task

CTR-GCN often does not directly use [T,J,C].

Many implementations convert the data into:

[N,C,T,V,M]

Where:

N = Batch Size

C = Coordinates

T = Frames

V = Joints (Vertices)

M = Number of Persons

Understanding this conversion is the key step between landmark vectors and CTR-GCN training.



https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker
https://arxiv.org/abs/2107.12213
