# Jetson Nano Computer Vision Projects

A collection of real-time computer vision applications built for the NVIDIA Jetson Nano, exploring different use cases from anomaly detection to drowsiness monitoring. Each implementation leverages different pretrained models and approaches to demonstrate various AI capabilities on edge devices.

## 🎯 Project Overview

This repo contains multiple branches, each implementing a different computer vision task optimized for the Jetson Nano. The projects showcase practical applications of deep learning models running efficiently on embedded hardware, making them suitable for real-world deployment scenarios.

All implementations use:
- **Hardware:** NVIDIA Jetson Nano
- **Framework:** JetPack SDK (jetson-inference, jetson-utils)
- **Models:** Pretrained networks (DetectNet, PoseNet, MobileNetV3)
- **Language:** Python 3

---

## 📂 Branch Structure

### 🚨 **DetectNet**
*Person detection with anomaly logging*

Uses SSD-MobileNet-v2 to detect people in frame and triggers anomalies when someone stays in view for more than 5 seconds. Great for monitoring restricted areas or tracking dwell times.

**Key Features:**
- Real-time person detection
- Configurable time threshold (default: 5 seconds)
- CSV logging with timestamps and confidence scores
- Visual alerts (bounding boxes turn red on anomaly)
- Live statistics overlay

**Use Case:** Security monitoring, occupancy tracking, restricted area surveillance

---

### 🧍‍♂️ **PoseNet_SocialDistancing**
*Social distancing violation detection*

Leverages PoseNet to track body keypoints (specifically hip positions) and calculates distances between people. Flags violations when individuals are closer than the defined threshold.

**How it Works:**
1. Detects multiple people using PoseNet
2. Calculates midpoint between left and right hips for each person
3. Measures distance between all detected individuals
4. Triggers violation alert if distance falls below threshold

**Use Case:** COVID-19 compliance monitoring, crowd management, workplace safety

---

### 📐 **PoseNet_rule_based**
*Rule-based drowsiness detection using facial keypoints*

Early implementation of drowsiness detection using geometric rules derived from facial keypoints. Uses ratios of ear-to-shoulder distances and head width to detect head tilts and nodding.

**Detection Logic:**
- Head tilt detection (left/right)
- Head nodding (forward tilt)
- Relative thresholds based on head dimensions
- Works at various distances from camera

**Note:** This was the first approach before moving to the ML-based classifier in `PoseNet_MobileNetv3-mini`.

---

### 🧠 **PoseNet_MobileNetv3-mini** *(Recommended)*
*Advanced drowsiness detection with deep learning*

The most sophisticated implementation combining PoseNet for face tracking with a fine-tuned MobileNetV3 classifier trained specifically on drowsy vs. alert eye states.

**Architecture:**
1. **PoseNet** tracks facial keypoints to locate eye regions
2. Extracts bounding box around both eyes (224x224 with padding)
3. **MobileNetV3-mini** classifier predicts drowsiness state
4. Tracks sustained drowsiness over time
5. Triggers visual/audio alerts on violation

**Training Approach:**
- Transfer learning from ImageNet weights
- Froze convolutional layers (feature extraction)
- Trained final classification layer on custom drowsiness dataset
- Runs in FP16 for faster inference on Jetson Nano

**Key Advantages over rule-based:**
- More robust to lighting conditions
- Better generalization across different people
- Learned features vs. hand-crafted rules
- Higher accuracy in challenging scenarios

**Use Case:** Driver monitoring systems, operator fatigue detection, safety applications

---

## 🚀 Getting Started

### Prerequisites
```bash
# Clone jetson-inference repository
git clone https://github.com/dusty-nv/jetson-inference.git
cd jetson-inference

# Launch Docker container (includes all dependencies)
docker/run.sh
```

### Running a Project

1. **Switch to desired branch:**
```bash
git checkout <branch-name>
```

2. **Run the inference script:**
```bash
# For DetectNet
python3 detectnet.py /dev/video0

# For PoseNet projects
python3 inference.py /dev/video0

# For CSI camera
python3 inference.py csi://0
```

3. **Exit:** Press `q` to quit

---

## 📊 Branch Comparison

| Branch | Model(s) | Detection Type | Logging | Training Required |
|--------|----------|----------------|---------|-------------------|
| **DetectNet** | SSD-MobileNet-v2 | Person presence | CSV | No |
| **PoseNet_SocialDistancing** | PoseNet | Distance between people | On-screen | No |
| **PoseNet_rule_based** | PoseNet | Head pose geometry | On-screen | No |
| **PoseNet_MobileNetv3-mini** | PoseNet + MobileNetV3 | Eye state (ML) | On-screen | Yes* |

*Pretrained models included in the repo

---

## 🔧 Technical Details

**Jetson Nano Optimizations:**
- FP16 (half-precision) inference for GPU acceleration
- TensorRT engine optimization
- Efficient memory management for embedded systems
- Real-time performance (15-30 FPS depending on model)

**Supported Cameras:**
- USB webcams (`/dev/video0`, `/dev/video1`, etc.)
- CSI cameras (`csi://0`)
- Video files for testing

---

## 💡 Future Improvements

- [ ] Audio alert integration for drowsiness detection
- [ ] Multi-camera support for social distancing
- [ ] Database logging instead of CSV
- [ ] Web dashboard for remote monitoring
- [ ] Model quantization for faster inference
- [ ] Mobile app integration

---

## 📝 Notes

The drowsiness detection project evolved through multiple iterations:
1. Started with rule-based geometric detection
2. Moved to ML-based classification for better accuracy
3. Fine-tuned on custom dataset for this specific use case

The training data and notebooks are included in the `PoseNet_MobileNetv3-mini` branch for anyone wanting to retrain or experiment with different architectures.

---

## 🙏 Acknowledgments

Built with NVIDIA's jetson-inference library and pretrained models from various sources. Training data for drowsiness detection was collected and labeled specifically for this project.
