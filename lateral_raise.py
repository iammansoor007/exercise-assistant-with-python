import cv2
import mediapipe as mp
import numpy as np
import share_state
from playsound import playsound
import threading
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
    
sound_playing = False
last_sound_time = 0  

# Global variable to store rep data
left_rep_data = {"left_counter": 0, "left_stage": None}
right_rep_data = {"right_counter": 0, "right_stage": None}

def get_reps():
    return left_rep_data,right_rep_data

def play_wrong_sound():
    global sound_playing, last_sound_time

    if sound_playing:
        return

    now = time.time()
    if now - last_sound_time < 2:  # Cooldown of 2 seconds
        return

    sound_playing = True
    last_sound_time = now

    try:
        playsound("static/wrong.mp3")
    finally:
        sound_playing = False

def calculate_angle(a,b,c):
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 

    radian = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radian * 180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle
    return angle


def process_video(cap):
    global left_rep_data    
    global right_rep_data      
    left_counter = 0
    right_counter = 0
    left_stage = None
    right_stage = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                break            

            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                if share_state.tracking_enabled:            
                    landmarks = results.pose_landmarks.landmark     
                       

                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y ]
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y ]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y ]

                    left_angle = calculate_angle(left_elbow,left_shoulder,left_hip)

                    if left_angle >=100 :
                        left_stage = "up"
                    if left_angle <30 and left_stage=="up":
                        left_stage ="down"
                        left_counter += 1 
                
                    left_rep_data["left_counter"] = left_counter
                    left_rep_data["left_stage"] = left_stage            

                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y ]
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y ]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y ]

                    right_angle = calculate_angle(right_elbow,right_shoulder,right_hip)

                    if  right_angle>=100 :
                        right_stage ="up"
                    if right_angle <30 and right_stage=="up":
                        right_stage = "down"
                        right_counter += 1
                    right_rep_data["right_counter"] = right_counter
                    right_rep_data["right_stage"] = right_stage    

                    if left_angle > 140 or right_angle > 140:
                        threading.Thread(target=play_wrong_sound,daemon=True).start()
                        text = "Shoulders too high"
                        font = cv2.FONT_HERSHEY_COMPLEX
                        font_scale = 2
                        thickness = 4
                        (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
                        _, image_width, _ = image.shape
                        x = (image_width - text_width) // 2
                        y = 140  
                        cv2.putText(image, text, (x, y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)                                
                            

                    mp_drawing.draw_landmarks(image, results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(247,117,66), thickness=7 , circle_radius=3),
                                            mp_drawing.DrawingSpec(color=(247,66,230), thickness=7 , circle_radius=3)
                                            )
            
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
