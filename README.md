# Drowsiness Detection System

Real-time driver drowsiness monitoring using PoseNet for face tracking and MobileNetV3 for eye state classification on Jetson Nano.

## How It Works

The system tracks a person's face using PoseNet to locate eye regions, then runs those regions through a trained MobileNetV3 model to detect drowsiness. If drowsiness is detected for more than 3 consecutive seconds, a violation is triggered with visual alerts.

## Project Structure

```
├── inference.py                    # Main detection script
├── posenet_pipeline.py             # PoseNet pipeline utilities
├── Models/
│   ├── MobileNet_224_FineTuned.pth
│   └── MobileNet_224_Scratch.pth
├── Notebooks/
│   ├── prepare_images.ipynb
│   ├── prepare_training_data.ipynb
│   ├── training_mobilenet_FineTuned.ipynb
│   └── training_mobilenet_Scratch.ipynb
└── TRAINING_DATA/
    ├── DROWSY/
    └── NOT_DROWSY/
```

## Quick Start

Run detection on default camera:
```bash
python3 inference.py /dev/video0
```

Run on CSI camera:
```bash
python3 inference.py csi://0
```

Press `q` to quit.

## Requirements

- Jetson Nano with JetPack 4.6+
- PyTorch
- jetson-inference
- jetson-utils
- OpenCV
- torchvision
- PIL

## Model Training

Training notebooks are in the `Notebooks/` directory:

1. `prepare_images.ipynb` - Image collection and preprocessing
2. `prepare_training_data.ipynb` - Dataset preparation
3. `training_mobilenet_FineTuned.ipynb` - Transfer learning approach
4. `training_mobilenet_Scratch.ipynb` - Training from scratch

The fine-tuned model is used by default in `inference.py`.

## Detection Features

- Real-time pose estimation with keypoint visualization
- Eye region detection and bounding boxes
- Binary classification (DROWSY_YES / DROWSY_NOT)
- 3-second violation threshold
- Visual alerts with flashing overlay
- Statistics panel showing:
  - Total violations
  - Current drowsy duration
  - Alert status
  - Time since last violation

## Thresholds

- Drowsiness probability: 0.8 (80%)
- Violation trigger: 3 seconds continuous drowsiness
- Model input: 224x224 RGB images

## Notes

The model runs in FP16 (half precision) on GPU for faster inference. Eye regions are extracted with padding to ensure both eyes are captured within the bounding box.
