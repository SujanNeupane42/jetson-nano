# Drowsiness Detection System

A real-time driver monitoring system that detects drowsiness using PoseNet for face tracking and MobileNetV3 for classification, optimized to run on Jetson Nano.

## How It Works

The system uses PoseNet to track facial keypoints and locate eye regions in real-time. When both eyes are detected, it extracts a bounding box around them and passes the cropped image (224x224 pixels with zero padding) to a MobileNetV3-mini classifier trained to distinguish between drowsy and alert states.

I used a transfer learning approach with a pretrained MobileNet-v3 model. The original ImageNet classifier was replaced with a 2-class output layer, and I froze the earlier convolutional layers to use them as a feature extractor. Only the final classification layer was trained on the drowsiness dataset, which helped get decent results without needing massive amounts of training data.

Here are some examples from the training set:

### Training Data Examples

**Non-Drowsy Sample**  
![Non-Drowsy training sample](Images/non-drowsy_training_image_sample.jpg)

**Drowsy Sample**  
![Drowsy training sample](Images/drowsy_training_image_sample.jpg)

Since the eye region images weren't always 224x224, I applied zero padding to standardize the input size for the model.

### Detection Output

Once the system is running, it provides real-time predictions. If someone is alert and attentive, you'll see something like this:

![No drowsiness detected](Images/no_drowsy.jpg)

When drowsiness is first detected:

![Drowsiness detected](Images/base_drowsy.jpg)

And if drowsiness persists for more than a few seconds, the system triggers an alert:

![Alert triggered](Images/alert_Drowsy.jpg)

The violation threshold is configurable—right now it's set to trigger after sustained drowsiness for N seconds. In a real deployment, this could easily be connected to an audio alarm to wake up the driver.


## Project Structure

```
├── inference.py                    # Main pipeline utilities
├── Models/
│   ├── MobileNet_224_FineTuned.pth
│   └── MobileNet_224_Scratch.pth
├── Notebooks/
│   ├── prepare_images.ipynb
│   ├── prepare_training_data.ipynb
│   ├── training_mobilenet_FineTuned.ipynb
│   └── training_mobilenet_Scratch.ipynb
```

## Getting Started

Clone the repo and run the inference script with your camera source:

**USB Camera:**
```bash
python3 inference.py /dev/video0
```

**CSI Camera:**
```bash
python3 inference.py csi://0
```

Press `q` to exit the application.

## Dependencies

- Jetson Nano (JetPack 4.6 or later)
- PyTorch
- jetson-inference
- jetson-utils
- OpenCV
- torchvision
- PIL

## Training Process

If you want to retrain the models or experiment with different architectures, check out the notebooks in `Notebooks/`:

1. **prepare_images.ipynb** – Collect and preprocess images
2. **prepare_training_data.ipynb** – Organize the dataset
3. **training_mobilenet_FineTuned.ipynb** – Fine-tune pretrained MobileNet (recommended)
4. **training_mobilenet_Scratch.ipynb** – Train from scratch

The fine-tuned model (`MobileNet_224_FineTuned.pth`) is what gets loaded by default in the inference script since it performed better with limited training data.

## Features

- Real-time pose estimation with facial keypoint tracking
- Automatic eye region detection and bounding box extraction
- Binary classification (drowsy vs. alert)
- Configurable violation threshold based on sustained drowsiness
- Visual alerts with screen overlay when drowsiness persists
- Live statistics display:
  - Total violation count
  - Current drowsy duration
  - Alert status
  - Time since last violation

## Configuration

**Key Parameters:**
- Drowsiness confidence threshold: 0.8 (80%)
- Violation trigger duration: N seconds of continuous drowsiness
- Input image size: 224x224 RGB

## Technical Notes

The model runs in FP16 (half precision) mode on the GPU to maximize inference speed on the Jetson Nano. Eye regions are extracted with padding applied to ensure both eyes fit cleanly within the bounding box before classification.
