import math

def calculate_distance(pt1, pt2):
    """Calculate Euclidean distance between two points"""
    if pt1 is None or pt2 is None:
        return None
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def detect_drowsiness_absolute(keypoint_dict):
    """
    Detect drowsiness using ABSOLUTE thresholds that are determined based on the relative position of ears and shoulders
    so that even if the person is far or camera is zoomed out, it works.
    """
    
    nose = keypoint_dict.get(0)
    left_eye = keypoint_dict.get(1)
    right_eye = keypoint_dict.get(2)
    left_ear = keypoint_dict.get(3)
    right_ear = keypoint_dict.get(4)
    left_shoulder = keypoint_dict.get(5)
    right_shoulder = keypoint_dict.get(6)
    neck = keypoint_dict.get(17)
    
    # Calculate head width
    head_width = None
    if left_ear and right_ear:
        head_width = calculate_distance(left_ear, right_ear)
    elif left_shoulder and right_shoulder:
        head_width = calculate_distance(left_shoulder, right_shoulder)
    
    if head_width is None or head_width < 10:  # if it too small leave it
        return False, "Insufficient keypoints", {}
    
    drowsy = False
    reasons = []
    metrics = {'head_width': head_width}
    
    # Case 1 & 2: Head Tilt (Left or Right)
    # When alert: ear-to-shoulder distance is ~1.2-1.5x head width
    # When tilted: ear-to-shoulder distance drops to ~0.5-0.8x head width
    
    if left_ear and left_shoulder:
        left_tilt_dist = calculate_distance(left_ear, left_shoulder)
        left_tilt_ratio = left_tilt_dist / head_width
        metrics['left_tilt_ratio'] = left_tilt_ratio
        
        # Threshold: If ratio < 0.7, head is tilted left
        if left_tilt_ratio < 0.7:
            drowsy = True
            reasons.append(f"Head TILTED LEFT (ratio: {left_tilt_ratio:.2f})")
    
    if right_ear and right_shoulder:
        right_tilt_dist = calculate_distance(right_ear, right_shoulder)
        right_tilt_ratio = right_tilt_dist / head_width
        metrics['right_tilt_ratio'] = right_tilt_ratio
        
        # Threshold: If ratio < 0.7, head is tilted right
        if right_tilt_ratio < 0.7:
            drowsy = True
            reasons.append(f"Head TILTED RIGHT (ratio: {right_tilt_ratio:.2f})")
    
    # Case 3: Head Drop Forward
    # When alert: nose-to-neck distance is ~0.8-1.2x head width
    # When dropped: nose-to-neck distance drops to ~0.3-0.5x head width
    
    if nose and neck:
        forward_dist = calculate_distance(nose, neck)
        forward_ratio = forward_dist / head_width
        metrics['forward_ratio'] = forward_ratio
        
        # Threshold: If ratio < 0.5, head dropped forward
        if forward_ratio < 0.5:
            drowsy = True
            reasons.append(f"Head DROPPED FORWARD (ratio: {forward_ratio:.2f})")
    
    # Case 4: Yawning (This is still beta and may not work at all)
    # When alert: eye-to-nose vertical distance is ~0.15-0.25x head width
    # When yawning: eye-to-nose distance increases to ~0.35-0.5x head width (mouth opens)
    
    if nose and left_eye and right_eye:
        avg_eye_y = (left_eye[1] + right_eye[1]) / 2
        eye_nose_dist = abs(nose[1] - avg_eye_y)
        yawn_ratio = eye_nose_dist / head_width
        metrics['yawn_ratio'] = yawn_ratio
        
        # Threshold: If ratio > 0.35, person is yawning
        if yawn_ratio > 0.35:
            drowsy = True
            reasons.append(f"YAWNING detected (ratio: {yawn_ratio:.2f})")
    
    reason = " | ".join(reasons) if reasons else "ALERT"
    
    return drowsy, reason, metrics