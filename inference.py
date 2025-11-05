#!/usr/bin/env python3

import cv2
import time
import sys
import argparse
import numpy as np
from jetson.utils import cudaToNumpy, cudaMemcpy, cudaFromNumpy
from jetson_inference import poseNet
from jetson_utils import videoSource, cudaImage, videoSource, videoOutput, Log
from jetson_inference import detectNet


parser = argparse.ArgumentParser(description="Social Distancing Detection using Pose Estimation.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("--output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="none", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.3, help="minimum detection threshold to use")
parser.add_argument("--distance-threshold", type=int, default=600, help="minimum pixel distance for social distancing (default: 25 pixels)")


try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

net = poseNet(args.network, sys.argv, args.threshold)

# this prints the keypoint names and their corresponding id
# num_keypoints = net.GetNumKeypoints()
# for idx in range(num_keypoints):
#     name = net.GetKeypointName(idx)
#     print(f"{idx}: {name}")

'''
0: nose
1: left_eye
2: right_eye
3: left_ear
4: right_ear
5: left_shoulder
6: right_shoulder
7: left_elbow
8: right_elbow
9: left_wrist
10: right_wrist
11: left_hip
12: right_hip
13: left_knee
14: right_knee
15: left_ankle
16: right_ankle
17: neck
'''
input = videoSource(args.input, argv=sys.argv)

KEYPOINT_LEFT_HIP = 11
KEYPOINT_RIGHT_HIP = 12

keypoint_color = (50, 205, 50)       # Green for keypoints
link_color = (255, 144, 30)          # Orange for skeleton links
safe_color = (0, 255, 0)             # Green for safe distance
violation_color = (0, 0, 255)        # Red for violations
label_text_color = (255, 255, 255)   # White for text
DISTANCE_THRESHOLD = args.distance_threshold


def calculate_euclidean_distance(point1, point2):
    """
    calculate euclidean distance between two points.
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def get_person_center(pose):
    """
    calculate the center point of a person using hip keypoints then return the midpoint of left and right hips of each person
    """
    left_hip = None
    right_hip = None
    
    for kp in pose.Keypoints:
        if int(kp.ID) == KEYPOINT_LEFT_HIP:
            left_hip = (int(kp.x), int(kp.y))
        elif int(kp.ID) == KEYPOINT_RIGHT_HIP:
            right_hip = (int(kp.x), int(kp.y))
    
    # getting the midpoint if only both hips are detected
    if left_hip is not None and right_hip is not None:
        center_x = (left_hip[0] + right_hip[0]) // 2
        center_y = (left_hip[1] + right_hip[1]) // 2
        return (center_x, center_y)
    
    return None


def get_person_bounding_box(pose):
    """
    calculate bounding box around a person based on all detected keypoints. uses padding to zoom out a bit
    """
    x_coords = []
    y_coords = []
    
    for kp in pose.Keypoints:
        x_coords.append(int(kp.x))
        y_coords.append(int(kp.y))
    
    if len(x_coords) == 0:
        return None
    
    # Add padding to bounding box
    padding = 20
    x1 = max(0, min(x_coords) - padding)
    y1 = max(0, min(y_coords) - padding)
    x2 = max(x_coords) + padding
    y2 = max(y_coords) + padding
    
    return (x1, y1, x2, y2)


def detect_social_distancing_violations(poses):
    """
    if distance between two people's hip midpoint is less than threshold, marking as violation
    """
    violations = []
    person_centers = []
    
    # getting centers for all detected people
    for pose in poses:
        center = get_person_center(pose)
        person_centers.append(center)
    
    num_people = len(poses)
    for i in range(num_people):
        for j in range(i + 1, num_people):
            center1 = person_centers[i]
            center2 = person_centers[j]

            # only calculate distance if both person's centers are available
            if center1 is None or center2 is None:
                continue
            
            distance = calculate_euclidean_distance(center1, center2)
            
            # if distance is below threshold;  this is social distancing voilation
            if distance < DISTANCE_THRESHOLD:
                violations.append((i, j, distance, center1, center2))
    
    return violations, person_centers


try:
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_count = 0
    total_violations = 0
    
    while True:
        img = input.Capture()
        
        if img is None:
            print("Failed to capture frame")
            continue
        
        frame_count += 1
        
        # Copy to CPU for OpenCV processing
        raw_img = cudaImage(width=img.width, height=img.height, format=img.format)
        cudaMemcpy(raw_img, img)
        frame = cudaToNumpy(raw_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        poses = net.Process(img, overlay='none')
        
        violations, person_centers = detect_social_distancing_violations(poses)
        
        violation_set = set()
        for violation in violations:
            violation_set.add(violation[0]) 
            violation_set.add(violation[1]) 
        
        for person_idx, pose in enumerate(poses):
            keypoint_dict = {}
            
            for kp in pose.Keypoints:
                kp_id = int(kp.ID)
                keypoint_dict[kp_id] = (int(kp.x), int(kp.y))
            
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
            
            for kp_id, (x, y) in keypoint_dict.items():
                cv2.circle(frame, (x, y), 6, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), 4, keypoint_color, -1)
            
            # drawing bounding box around each person
            bbox = get_person_bounding_box(pose)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                
                if person_idx in violation_set:
                    bbox_color = violation_color
                    label = f"Person {person_idx + 1} - VIOLATION"
                else:
                    bbox_color = safe_color
                    label = f"Person {person_idx + 1} - Safe"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                
                font_scale = 0.6
                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 2)
                label_x = x1
                label_y = max(y1 - 8, 15)
                
                cv2.rectangle(frame, (label_x, label_y - text_h - 6), (label_x + text_w + 8, label_y + 2), bbox_color, -1)
                cv2.putText(frame, label, (label_x + 4, label_y - 2), font, font_scale, label_text_color, 2, cv2.LINE_AA)
                
                center = person_centers[person_idx]
                if center is not None:
                    cv2.circle(frame, center, 8, bbox_color, -1)
                    cv2.circle(frame, center, 10, (255, 255, 255), 2)
        
        for person1_idx, person2_idx, distance, center1, center2 in violations:
            cv2.line(frame, center1, center2, violation_color, 3)
            mid_x = (center1[0] + center2[0]) // 2
            mid_y = (center1[1] + center2[1]) // 2
            distance_label = f"{int(distance)}px"
            font_scale = 0.7
            (text_w, text_h), _ = cv2.getTextSize(distance_label, font, font_scale, 2)
            
            cv2.rectangle(frame, (mid_x - text_w // 2 - 5, mid_y - text_h - 5), (mid_x + text_w // 2 + 5, mid_y + 5), (0, 0, 0), -1)
            cv2.rectangle(frame, (mid_x - text_w // 2 - 5, mid_y - text_h - 5), (mid_x + text_w // 2 + 5, mid_y + 5), violation_color, 2)
            cv2.putText(frame, distance_label, (mid_x - text_w // 2, mid_y - 5), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
        
        # this is our violation statistics panel
        num_people = len(poses)
        num_violations = len(violations)
        total_violations += num_violations
        
        panel_height = 120
        panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # dark gray background for the panel
        
        y_offset = 30
        cv2.putText(panel, f"People Detected: {num_people}", (20, y_offset), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        y_offset += 35
        violation_color_text = violation_color if num_violations > 0 else safe_color
        cv2.putText(panel, f"Active Violations: {num_violations}", (20, y_offset), font, 0.7, violation_color_text, 2, cv2.LINE_AA)
        
        y_offset += 35
        cv2.putText(panel, f"Distance Threshold: {DISTANCE_THRESHOLD}px", (20, y_offset), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        
        # putting the stats panel and frame together
        combined_frame = np.vstack([panel, frame])
        cv2.imshow("Social Distancing Detection - PoseNet", combined_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if not input.IsStreaming():
            break

        
except KeyboardInterrupt:
    print("ending detection")
finally:
    cv2.destroyAllWindows()
