#!/usr/bin/env python3

import cv2
import time
import sys
import cv2
import argparse
from jetson.utils import cudaToNumpy, cudaMemcpy, cudaFromNumpy
from jetson_inference import poseNet
from jetson_utils import videoSource, cudaImage, videoSource, videoOutput, Log
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from PIL import Image
from jetson_inference import detectNet
from datetime import datetime


parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("--output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="none", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.3, help="minimum detection threshold to use") 


try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

net = poseNet(args.network, sys.argv, args.threshold)

# create video source
input = videoSource(args.input, argv=sys.argv)

mappings = {
    0: "Nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "Left Ear",
    4: "Right Ear",
    5: "Left shoulder",
    6: "Right Shoulder",
    17: "neck"
}

# Colors
keypoint_color = (50, 205, 50)  
link_color = (255, 144, 30)     
head_bbox_color = (0, 191, 255)  
eye_bbox_color = (255, 0, 255)   
label_text_color = (255, 255, 255)

filtered_keypoints_ids = set(mappings.keys())

print("PoseNet model loaded successfully!")
print("Run the next cell to start live detection with bounding boxes")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = mobilenet_v3_small(weights=False)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
model.load_state_dict(torch.load("Models/MobileNet_224_FineTuned.pth"))
model.to(device)

# Convert model to half precision (FP16) for faster inference
if device.type == 'cuda':
    model = model.half()
    print("Model converted to FP16 (half precision) for optimized inference")
else:
    print("Running on CPU - keeping FP32 precision")

model.eval()

eye_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['DROWSY_NOT', 'DROWSY_YES']
print("MobileNet Drowsiness Detection Model loaded successfully!")
print(f"Device: {device}")
print(f"Classes: {class_names}")

# number of seconds person must be drowsy to trigger violation
DROWSY_THRESHOLD_SECONDS = 3  
drowsy_start_time = None 
drowsy_duration = 0.0  # 

# drowsiness violation statistics
total_violations = 0
last_violation_time = None
in_violation_state = False

print(f"\nDrowsiness Monitoring Settings:")
print(f"- Violation Threshold: {DROWSY_THRESHOLD_SECONDS} seconds")
print(f"- Drowsiness threshold probability: 0.8")

try:
    font = cv2.FONT_HERSHEY_SIMPLEX    
    while True:
        img = input.Capture()
        
        if img is None:
            print("Failed to capture frame")
            continue
        
        # copy to CPU for OpenCV processing
        raw_img = cudaImage(width=img.width, height=img.height, format=img.format)
        cudaMemcpy(raw_img, img)
        frame = cudaToNumpy(raw_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        poses = net.Process(img, overlay='none')
        
        for obj_idx, pose in enumerate(poses):
            keypoint_dict = {}
            
            for kp in pose.Keypoints:
                kp_id = int(kp.ID)
                if kp_id in filtered_keypoints_ids:
                    keypoint_dict[kp_id] = (int(kp.x), int(kp.y))
            
            # get key points for eyes
            left_eye = keypoint_dict.get(1)
            right_eye = keypoint_dict.get(2)
            
            # draw eye bounding box if both eyes are detected
            if left_eye is not None and right_eye is not None:
                eye_x_coords = [left_eye[0], right_eye[0]]
                eye_y_coords = [left_eye[1], right_eye[1]]
                
                eye_min_x = min(eye_x_coords)
                eye_max_x = max(eye_x_coords)
                eye_min_y = min(eye_y_coords)
                eye_max_y = max(eye_y_coords)
                
                # add padding around eyes
                eye_width = eye_max_x - eye_min_x
                eye_height = eye_max_y - eye_min_y
                
                # making the box wider and taller to cover both eyes comfortably inside the box
                eye_padding_x = int(eye_width * 0.5)
                eye_padding_y = int(max(eye_height * 2.0, eye_width * 0.4))  # just making sure it tall enough
                
                eye_bbox_x1 = max(0, eye_min_x - eye_padding_x)
                eye_bbox_y1 = max(0, eye_min_y - eye_padding_y)
                eye_bbox_x2 = min(frame.shape[1], eye_max_x + eye_padding_x)
                eye_bbox_y2 = min(frame.shape[0], eye_max_y + eye_padding_y)
                
                # extract eye region for drowsiness detection
                eye_region = frame[eye_bbox_y1:eye_bbox_y2, eye_bbox_x1:eye_bbox_x2]
                
                # perform drowsiness detection if eye region is valid
                drowsy_label = "Unknown"
                drowsy_prob = 0.0
                if eye_region.shape[0] > 0 and eye_region.shape[1] > 0:
                    try:
                        # convert BGR to RGB for PIL
                        eye_region_rgb = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
                        eye_pil = Image.fromarray(eye_region_rgb)
                        
                        # apply transforms
                        eye_tensor = eye_transform(eye_pil).unsqueeze(0).to(device)
                        
                        # convert to half precision if model is in half precision
                        if device.type == 'cuda':
                            eye_tensor = eye_tensor.half()
                        
                        # predict
                        with torch.no_grad():
                            output = model(eye_tensor)
                            prob = torch.sigmoid(output).item()
                            pred_class = int(prob > 0.8)  # only classify as drowsy if probability > 0.8
                            drowsy_label = class_names[pred_class]
                            drowsy_prob = prob if pred_class == 1 else (1 - prob)
                    except Exception as e:
                        print(f"Error in drowsiness detection: {e}")
                
                # set color based on drowsiness state
                if drowsy_label == "DROWSY_YES":
                    bbox_color = (0, 0, 255)  # red for drowsy
                    label_bg_color = (0, 0, 255)
                    
                    # start time tracking if drowsiness is detected
                    if drowsy_start_time is None:
                        drowsy_start_time = time.time()
                    
                    # this calculates the number of seconds its been since the person started being drowsy
                    drowsy_duration = time.time() - drowsy_start_time
                    
                    # if threshold is exceeded, we give drowsines violation alarm
                    if drowsy_duration >= DROWSY_THRESHOLD_SECONDS and not in_violation_state:
                        in_violation_state = True
                        total_violations += 1
                        last_violation_time = datetime.now()
                        print(f"\n⚠️ DROWSINESS VIOLATION DETECTED! (Violation #{total_violations})")
                        print(f"Time: {last_violation_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"Duration: {drowsy_duration:.1f} seconds")
                        
                elif drowsy_label == "DROWSY_NOT":
                    bbox_color = (0, 255, 0) 
                    label_bg_color = (0, 255, 0)
                    
                    # one detection is stopped, reseting the drowsy tracking variables
                    if drowsy_start_time is not None:
                        if in_violation_state:
                            print(f"   Violation ended. Total duration: {drowsy_duration:.1f} seconds")
                        drowsy_start_time = None
                        drowsy_duration = 0.0
                        in_violation_state = False
                else:
                    bbox_color = eye_bbox_color  
                    label_bg_color = eye_bbox_color
                
                # draw bounding box with color based on drowsiness state
                cv2.rectangle(frame, (eye_bbox_x1, eye_bbox_y1), (eye_bbox_x2, eye_bbox_y2), bbox_color, 2)
                
                # prepare label with drowsiness info
                eye_label = f"{drowsy_label} ({drowsy_prob*100:.1f}%)"
                eye_font_scale = 0.6
                (eye_text_w, eye_text_h), _ = cv2.getTextSize(eye_label, font, eye_font_scale, 2)
                
                eye_label_x = eye_bbox_x1
                eye_label_y = max(eye_bbox_y1 - 8, 15)
                
                # draw label background and text
                cv2.rectangle(frame, (eye_label_x, eye_label_y - eye_text_h - 6), (eye_label_x + eye_text_w + 8, eye_label_y + 2), label_bg_color, -1)
                cv2.putText(frame, eye_label, (eye_label_x + 4, eye_label_y - 2), font, eye_font_scale, label_text_color, 2, cv2.LINE_AA)
            
            # draw skeleton links
            try:
                for link in pose.Links:
                    if isinstance(link, tuple) and len(link) == 2:
                        start_id, end_id = link
                    else:
                        continue
                    
                    if start_id in keypoint_dict and end_id in keypoint_dict:
                        start_point = keypoint_dict[start_id]
                        end_point = keypoint_dict[end_id]
                        cv2.line(frame, start_point, end_point, link_color, 2)
            except:
                pass
            
            # draw keypoints
            for kp_id, (x, y) in keypoint_dict.items():
                cv2.circle(frame, (x, y), 6, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), 4, keypoint_color, -1)
        
        # drawing an alarm overlay if in drowsiness violation state
        if in_violation_state:
            if int(time.time() * 2) % 2 == 0:  # Flash every 0.5 seconds
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            
            alarm_text = "⚠️ DROWSINESS ALERT! ⚠️"
            alarm_font_scale = 1.2
            alarm_thickness = 3
            (alarm_w, alarm_h), _ = cv2.getTextSize(alarm_text, font, alarm_font_scale, alarm_thickness)
            alarm_x = (frame.shape[1] - alarm_w) // 2
            alarm_y = 60
            
            cv2.rectangle(frame, (alarm_x - 10, alarm_y - alarm_h - 10), (alarm_x + alarm_w + 10, alarm_y + 10), (0, 0, 0), -1)
            cv2.putText(frame, alarm_text, (alarm_x, alarm_y), font, alarm_font_scale, (0, 0, 255), alarm_thickness, cv2.LINE_AA)
        
        # violation statistics panel
        panel_x = 10
        panel_y = 30
        panel_padding = 10
        line_height = 30
        panel_font_scale = 0.6
        panel_thickness = 2
        
        stats_lines = []
        stats_lines.append(f"Total Violations: {total_violations}")
        stats_lines.append(f"Drowsy Duration: {drowsy_duration:.1f}s / {DROWSY_THRESHOLD_SECONDS}s")
        
        if in_violation_state:
            status_text = "Status: VIOLATION"
            status_color = (0, 0, 255)
        elif drowsy_start_time is not None:
            status_text = "Status: DROWSY"
            status_color = (255, 165, 0)
        else:
            status_text = "Status: ALERT"
            status_color = (0, 255, 0)
        stats_lines.append(status_text)
        
        # calculating the number of seconds its been since last violation was detected
        if last_violation_time:
            time_since_violation = (datetime.now() - last_violation_time).total_seconds()
            if time_since_violation < 60:
                stats_lines.append(f"Last Violation: {time_since_violation:.0f}s ago")
            elif time_since_violation < 3600:
                stats_lines.append(f"Last Violation: {time_since_violation/60:.1f}m ago")
            else:
                stats_lines.append(f"Last Violation: {last_violation_time.strftime('%H:%M:%S')}")
        else:
            stats_lines.append("Last Violation: None")
        
        max_text_width = 0
        for line in stats_lines:
            (text_w, text_h), _ = cv2.getTextSize(line, font, panel_font_scale, panel_thickness)
            max_text_width = max(max_text_width, text_w)
        
        panel_width = max_text_width + 2 * panel_padding
        panel_height = len(stats_lines) * line_height + 2 * panel_padding
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y - 20), 
                     (panel_x + panel_width, panel_y + panel_height - 20), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        for i, line in enumerate(stats_lines):
            y_pos = panel_y + i * line_height
            if "Status:" in line:
                cv2.putText(frame, line, (panel_x + panel_padding, y_pos), font, panel_font_scale, status_color, panel_thickness, cv2.LINE_AA)
            else:
                cv2.putText(frame, line, (panel_x + panel_padding, y_pos), font, panel_font_scale, (255, 255, 255), panel_thickness, cv2.LINE_AA)
        
        cv2.imshow("Drowsiness Detection with PoseNet", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if not input.IsStreaming():
            break

        
except KeyboardInterrupt:
    print("\nPoseNet detection stopped by user")
finally:
    cv2.destroyAllWindows()