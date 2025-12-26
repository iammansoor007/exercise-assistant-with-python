import cv2
import mediapipe as mp
import numpy as np
import share_state

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Global variable to store rep data
left_rep_data = {"left_counter": 0, "left_stage": None}
right_rep_data = {"right_counter": 0, "right_stage": None}


def get_reps():
    return left_rep_data,right_rep_data

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def process_video(cap):
    global left_rep_data      
    global right_rep_data   
    left_counter, right_counter = 0, 0
    left_stage, right_stage = None, None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():            
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                if share_state.tracking_enabled:
                    landmarks = results.pose_landmarks.landmark

                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    if left_angle > 160:
                        left_stage = "Down"
                    if left_angle < 30 and left_stage == "Down":
                        left_stage = "Up"
                        left_counter += 1

                    left_rep_data["left_counter"] = left_counter
                    left_rep_data["left_stage"] = left_stage


                    if right_angle > 160:
                        right_stage = "Down"
                    if right_angle < 30 and right_stage == "Down":
                        right_stage = "Up"
                        right_counter += 1

                    right_rep_data["right_counter"] = right_counter
                    right_rep_data["right_stage"] = right_stage


                    mp_drawing.draw_landmarks(image, results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(247,117,66), thickness=7 , circle_radius=3),
                                        mp_drawing.DrawingSpec(color=(247,66,230), thickness=7 , circle_radius=3)
                                        )                
    
            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
