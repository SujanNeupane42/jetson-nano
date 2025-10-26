#!/usr/bin/env python3
import sys
import argparse
import cv2
from jetson.utils import cudaToNumpy, cudaMemcpy
from jetson_inference import poseNet
from jetson_utils import videoSource, Log, cudaImage
from drowsiness_detector import detect_drowsiness_absolute

# Keypoint mappings for face/head detection
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

parser = argparse.ArgumentParser(description="Run pose estimation DNN with drowsiness detection.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=poseNet.Usage() + videoSource.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="none", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

net = poseNet(args.network, sys.argv, args.threshold)
input = videoSource(args.input, argv=sys.argv)

keypoint_color = (50, 205, 50)  # Lime green for keypoints
link_color = (255, 144, 30)     # Bright orange for skeleton links
alert_color = (0, 191, 255)     # Deep sky blue for alert state
drowsy_color = (0, 0, 255)      # Red for drowsy state
label_text_color = (255, 255, 255)  # White text

filtered_keypoints_ids = set(mappings.keys())

while True:
    img = input.Capture()

    if img is None: 
        continue  
    
    # looks like the jetson-inference package automatically makes changes to the original image feed in gpu like -
    # - adding keypoints and their connections. I want to manually make change to the frame in the video feed. Hence,
    # - making copy of the frame in GPU (by converting it to NumPy and to OpenCV format), I get to do any custom manipulaton I want in original frame
    raw_img = cudaImage(width=img.width, height=img.height, format=img.format)
    cudaMemcpy(raw_img, img) # copies from Cuda to cpu
    frame = cudaToNumpy(raw_img)  
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    
    poses = net.Process(img, overlay=args.overlay)

    # each detected pose is from a new person; hence, can also be used to do person/face detection
    for obj_idx, pose in enumerate(poses):
        keypoint_dict = {}
        for kp in pose.Keypoints:
            kp_id = int(kp.ID) # the category of keypoint
            if kp_id in filtered_keypoints_ids:
                keypoint_dict[kp_id] = (int(kp.x), int(kp.y))
        
        is_drowsy, reason, metrics = detect_drowsiness_absolute(keypoint_dict)
        bbox_color = drowsy_color if is_drowsy else alert_color
        label_bg_color = bbox_color
        
        # Print drowsiness status
        status = "DROWSY" if is_drowsy else "ALERT"
        
        # If i have at least 3 keypoints cordinates among left/right ears and shoulders, i can zoom out a bit and also do object/face detection
        # and add a bounding box here.
        # 3 = Left Ear, 4 = Right Ear, 5 = Left Shoulder, 6 = Right Shoulder
        left_ear = keypoint_dict.get(3)
        right_ear = keypoint_dict.get(4)
        left_shoulder = keypoint_dict.get(5)
        right_shoulder = keypoint_dict.get(6)
        
        available = [left_ear, right_ear, left_shoulder, right_shoulder]
        available_count = sum(1 for pt in available if pt is not None)
        
        if available_count >= 3:

            # We have at least 3 points - estimate the 4th if needed using the x and y cordinates from the rest 3
            if left_ear is None and right_ear and left_shoulder and right_shoulder:
                width = right_shoulder[0] - right_ear[0]
                left_ear = (left_shoulder[0] - width, right_ear[1])
                
            elif right_ear is None and left_ear and left_shoulder and right_shoulder:
                width = left_ear[0] - left_shoulder[0]
                right_ear = (right_shoulder[0] + width, left_ear[1])
                
            elif left_shoulder is None and left_ear and right_ear and right_shoulder:
                width = left_ear[0] - right_ear[0]
                left_shoulder = (left_ear[0] - width, right_shoulder[1])
                
            elif right_shoulder is None and left_ear and right_ear and left_shoulder:
                width = right_ear[0] - left_ear[0]
                right_shoulder = (right_ear[0] + width, left_shoulder[1])
            
            # Calculate bounding box
            all_points = [left_ear, right_ear, left_shoulder, right_shoulder]
            x_coords = [pt[0] for pt in all_points]
            y_coords = [pt[1] for pt in all_points]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            width = max_x - min_x
            height = max_y - min_y
            
            # adding some padding to zoom out a bit otherwise the bounding box would contain face inside the shoulders and ears 
            # which wouldn't look too good.
            padding_x = int(width * 0.4)
            padding_y_top = int(height * 1.15)
            padding_y_bottom = int(height * 0.3)
            
            bbox_x1 = max(0, min_x - padding_x)
            bbox_y1 = max(0, min_y - padding_y_top)
            bbox_x2 = min(frame.shape[1], max_x + padding_x)
            bbox_y2 = min(frame.shape[0], max_y + padding_y_bottom)
            
            cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), bbox_color, 3)
            label = f"Person {obj_idx + 1}: {status}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            label_x = bbox_x1
            label_y = max(bbox_y1 - 15, 25)
            bg_x1 = label_x
            bg_y1 = label_y - text_height - 8
            bg_x2 = label_x + text_width + 10
            bg_y2 = label_y + 5
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), label_bg_color, -1)
            
            cv2.putText(frame, label, (label_x + 5, label_y), font, font_scale, label_text_color, font_thickness, cv2.LINE_AA)
            
            # Add drowsiness reason below if drowsy
            if is_drowsy and reason != "ALERT":
                reason_y = bbox_y2 + 25
                reason_font_scale = 0.5
                max_reason_width = bbox_x2 - bbox_x1
                
                cv2.putText(frame, reason, (bbox_x1, reason_y), cv2.FONT_HERSHEY_SIMPLEX, reason_font_scale, drowsy_color, 2, cv2.LINE_AA)
        
        
        # if only two cordinates are available among left/right ears and shoulders, we can still do drowsiness detection
        # but the bounding box is going to be pretty bad. in that case, i just use the min and max values of (x1, y1) and (x2, y2)
        # to estimate (x3, y3) and (x4, y4)
        elif available_count == 2:
            bbox_points = [pt for pt in available if pt is not None]
            x_coords = [pt[0] for pt in bbox_points]
            y_coords = [pt[1] for pt in bbox_points]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            width = max_x - min_x
            height = max_y - min_y
            
            # Estimate if too narrow
            if width < height * 0.3:
                estimated_width = int(height * 0.8)
                center_x = (min_x + max_x) // 2
                min_x = center_x - estimated_width // 2
                max_x = center_x + estimated_width // 2
                width = max_x - min_x
            
            padding_x = int(width * 0.4)
            padding_y_top = int(height * 1.15)
            padding_y_bottom = int(height * 0.3)
            
            bbox_x1 = max(0, min_x - padding_x)
            bbox_y1 = max(0, min_y - padding_y_top)
            bbox_x2 = min(frame.shape[1], max_x + padding_x)
            bbox_y2 = min(frame.shape[0], max_y + padding_y_bottom)
            
            cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), bbox_color, 3)
            label = f"Person {obj_idx + 1}: {status}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            label_x = bbox_x1
            label_y = max(bbox_y1 - 15, 25)            
            bg_x1 = label_x
            bg_y1 = label_y - text_height - 8
            bg_x2 = label_x + text_width + 10
            bg_y2 = label_y + 5

            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), label_bg_color, -1)
            cv2.putText(frame, label, (label_x + 5, label_y),  font, font_scale, label_text_color, font_thickness, cv2.LINE_AA)
            
            if is_drowsy and reason != "ALERT":
                reason_y = bbox_y2 + 25
                cv2.putText(frame, reason, (bbox_x1, reason_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, drowsy_color, 2, cv2.LINE_AA)
        
        # drawing the skeletol links by connecting the keypoints
        try:
            for link in pose.Links:
                if isinstance(link, tuple) and len(link) == 2:
                    start_id, end_id = link
                else:
                    continue
                
                if start_id in keypoint_dict and end_id in keypoint_dict:
                    start_point = keypoint_dict[start_id]
                    end_point = keypoint_dict[end_id]
                    cv2.line(frame, start_point, end_point, link_color, 3)
        except:
            pass
        
        for kp_id, (x, y) in keypoint_dict.items():
            cv2.circle(frame, (x, y), 7, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), 5, keypoint_color, -1)

    cv2.imshow("Drowsiness Detection with PoseNet", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()