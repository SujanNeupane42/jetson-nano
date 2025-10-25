#!/usr/bin/env python3

'''
This model will  be utilized for anamoly detection
Here, we can consider a hypothetical scenario where this system is implemented in a security system.
If it detects a person inside a safe/room/warehouse for more than 5 seconds, we consider it an anomaly and it will 
- write to a text file about the anomaly along with timestamp.

If no person detected, its okay. The anomaly warning only is triggered when a person is detected for more than 5 seconds
'''

import sys
import argparse
import cv2
import time
import csv
from datetime import datetime
from jetson_utils import videoSource, Log, cudaImage
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log
from jetson.utils import cudaToNumpy, cudaMemcpy, cudaFromNumpy


# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.6, help="minimum detection threshold to use") 

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# create video sources and outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)
	
# load the object detection network
net = detectNet(args.network, sys.argv, args.threshold)

# variables for tracking person detection
person_detected_start_time = None
person_detected = False
anomaly_logged = False
PERSON_CLASS_ID = 1  
DETECTION_THRESHOLD = 5.0 # number of seconds

# vsariables to track anomalies for the statistics panel
anomaly_count = 0
last_anomaly_time = None
last_anomaly_timestamp = "None"

# this is the csv file containing the logging info
csv_filename = "person_detection_log.csv"

try:
    # if file does not exist, create it and write the header
    with open(csv_filename, 'x', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Timestamp', 'Event', 'Duration (seconds)', 'Confidence'])
except FileExistsError:
    pass  


while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  
        
    raw_img = cudaImage(width=img.width, height=img.height, format=img.format)
    cudaMemcpy(raw_img, img) # copies from Cuda to cpu; i personally feel more comfortable applying transformations in cpu
    frame = cudaToNumpy(raw_img)   # creates a numpy array from the cuda image
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR) # convert to opencv frame to be manipulated later

    # detect objects/persons in the image; no overlay required as we will extract bounding boxes and overlay ourselves
    detections = net.Detect(img, overlay='none')

    # only get detected predictions for Class 1 or Person
    person_detections = [d for d in detections if d.ClassID == PERSON_CLASS_ID]

    # this will act as a counter; if a person is detected more than $DETECTION_THRESHOLD, trigger anomaly
    current_time = time.time()
    
    if len(person_detections) > 0:
        # if detected for first time, start the timer
        if person_detected_start_time is None:
            person_detected_start_time = current_time
            person_detected = True

            '''
            I also use this additional variable to track if an anomaly has been logged and reset the timer. If a person has been detected at 5th second,
            - the system will append a log to a csv file but if that object is continuted being detected beyond time, it will write log for each of the newer frames.
            So, i just reset it to prevent continuous logging as these would just be duplicates. 
            This would just create 100s of duplicate detected entries to my csv file even for just one single continuous detection.
            '''
            anomaly_logged = False
            print("Person detected - starting timer")
        
        # calculate how long person has been detected and if it is greater than $DETECTION_THRESHOLD, we will trigger the anomaly
        detection_duration = current_time - person_detected_start_time

        # check if person has been detected for more than 5 seconds; using this boolean to decide whether we will trigger anomaly case or not
        is_anomaly = detection_duration >= DETECTION_THRESHOLD

        # log to CSV here if we have anomaly detected for more than $DETECTION_THRESHOLD seconds
        if is_anomaly and not anomaly_logged:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S') # full timestamp
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for detection in person_detections:
                    writer.writerow([timestamp, 'ANOMALY: Person detected for >5 seconds', f'{detection_duration:.2f}', f'{detection.Confidence:.2f}'])                
            anomaly_logged = True

            # these will be recorded for the anomaly stats panel 
            anomaly_count += 1  # increase the anomaly counter
            last_anomaly_time = current_time  # record when this anomaly occurred
            last_anomaly_timestamp = timestamp  # saving formatted timestamp
            print(f"ANOMALY LOGGED: Person detected for {detection_duration:.2f} seconds")
        
        # iterating through person detections
        for detection in person_detections:
            
            # extracting bounding boxes for each detected person
            left = int(detection.Left)
            top = int(detection.Top)
            right = int(detection.Right)
            bottom = int(detection.Bottom)

            label_text = f"Person Detected | ({detection.Confidence:.2f}) | {detection_duration:.1f}s"
            
            # include duration in the label text itself
            if is_anomaly:
                box_color = (0, 0, 255)  # red color text for anomaly
                label_bg_color = (0, 0, 255)
            else:
                box_color = (0, 255, 0)  # green for normal detection; will switch to red after person is detected for more than 5 seconds
                label_bg_color = (0, 255, 0)
            
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            
            # prepare label with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

            # drawing label background to put text in its own box above the original bounding box
            label_y = max(top - 10, text_height + 10)
            cv2.rectangle(frame, (left, label_y - text_height - 5), (left + text_width + 10, label_y + 5), label_bg_color, -1)
            
            # put the text (anomaly or not anomaly) to the frame - now includes duration
            cv2.putText(frame,  label_text,  (left + 5, label_y), font, font_scale, (255, 255, 255),  font_thickness, cv2.LINE_AA)
        
        print(f"Detected {len(person_detections)} person(s) - Duration: {detection_duration:.2f}s")
    
    else:
        # if no person is detected, we just reset the time and anomalies will only be logged if the $DETECTION_THRESHOLD is met again
        if person_detected_start_time is not None:
            print("Person no longer detected - resetting timer")

        person_detected_start_time = None
        person_detected = False
        anomaly_logged = False

    # this will draw statistics panel in top-middle
    panel_width = 350
    panel_height = 140
    panel_x = (frame.shape[1] - panel_width) // 2  # top middle posiiton for the panel
    panel_y = 10 
    
    # background for panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # draw panel border
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)
    
    # panel title which contains anomaly stats
    title_font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "ANOMALY STATISTICS", (panel_x + 10, panel_y + 25), title_font, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    
    # anomaly count from in-memory variable
    count_text = f"Total Anomalies: {anomaly_count}"
    cv2.putText(frame, count_text, (panel_x + 10, panel_y + 50), title_font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    
    # anomaly type
    type_text = f"Type: Person >5s"
    cv2.putText(frame, type_text, (panel_x + 10, panel_y + 75), title_font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    
    # last anomaly timestamp from in-memory variable
    last_detected_text = f"Last: {last_anomaly_timestamp}"
    cv2.putText(frame, last_detected_text, (panel_x + 10, panel_y + 100), title_font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    # time since last anomaly (in seconds, minutes, or hours)
    if last_anomaly_time is not None:
        time_since = current_time - last_anomaly_time
        if time_since < 60:
            time_since_text = f"Time Since: {time_since:.1f}s ago"
        elif time_since < 3600:
            time_since_text = f"Time Since: {time_since/60:.1f}m ago"
        else:
            time_since_text = f"Time Since: {time_since/3600:.1f}h ago"
    else:
        time_since_text = "Time Since: N/A"
    
    cv2.putText(frame, time_since_text, (panel_x + 10, panel_y + 125), title_font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # rendering the frame using jetson utils
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cuda_frame = cudaFromNumpy(frame) # copying the opencv cpu frame to CUDA to render using jetson-utils's builtin function
    output.Render(cuda_frame)

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break