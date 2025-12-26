from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session, g
import cv2
import sqlite3
import hashlib
import os
from datetime import datetime, timedelta
import time
from functools import wraps
import numpy as np
import threading
import mediapipe as mp
import warnings
import logging
from contextlib import contextmanager
import atexit
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'ai-fitness-trainer-secret-key-2024-dev-only')
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_REFRESH_EACH_REQUEST'] = True

# Mediapipe initialization
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Try to import playsound, but continue without it if not available
SOUND_ENABLED = False
try:
    import playsound
    SOUND_ENABLED = True
    logger.info("🔊 Sound enabled")
except ImportError:
    logger.warning("🔇 Sound disabled - playsound module not found")

# ============ SOUND MANAGEMENT ============

sound_playing = False
last_sound_time = 0

def play_wrong_sound():
    """Play sound for incorrect form with cooldown"""
    global sound_playing, last_sound_time
    
    if not SOUND_ENABLED or sound_playing:
        return
    
    now = time.time()
    if now - last_sound_time < 2:  # Cooldown of 2 seconds
        return
    
    sound_playing = True
    last_sound_time = now
    
    try:
        playsound.playsound("static/wrong.mp3")
    except Exception as e:
        logger.error(f"Error playing sound: {e}")
    finally:
        sound_playing = False

# ============ UTILITY FUNCTIONS ============

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def check_visibility(landmark, threshold=0.5):
    """Check if landmark is visible enough"""
    return hasattr(landmark, 'visibility') and landmark.visibility > threshold

# ============ EXERCISE PROCESSORS ============

class ExerciseProcessors:
    """All exercise processors with biomechanically accurate calculations"""
    
    @staticmethod
    def bicep_curl(landmarks):
        """Bicep curl processor - Biomechanically accurate"""
        try:
            # Get landmarks
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            # Check visibility
            if not all([check_visibility(left_shoulder), check_visibility(left_elbow), check_visibility(left_wrist),
                       check_visibility(right_shoulder), check_visibility(right_elbow), check_visibility(right_wrist)]):
                return
            
            # Prepare coordinates
            left_coords = [
                [left_shoulder.x, left_shoulder.y],
                [left_elbow.x, left_elbow.y],
                [left_wrist.x, left_wrist.y]
            ]
            
            right_coords = [
                [right_shoulder.x, right_shoulder.y],
                [right_elbow.x, right_elbow.y],
                [right_wrist.x, right_wrist.y]
            ]
            
            # Calculate angles
            left_angle = calculate_angle(left_coords[0], left_coords[1], left_coords[2])
            right_angle = calculate_angle(right_coords[0], right_coords[1], right_coords[2])
            
            # Biomechanical thresholds for bicep curl
            EXTEND_THRESHOLD = 160  # Arm fully extended
            FLEX_THRESHOLD = 40     # Arm fully flexed
            
            # Get current states
            left_stage = share_state.reps_data['left_stage']
            right_stage = share_state.reps_data['right_stage']
            
            # State machine for left arm
            if left_angle > EXTEND_THRESHOLD:
                if left_stage != "Extended":
                    share_state.reps_data['left_stage'] = "Extended"
            elif left_angle < FLEX_THRESHOLD and left_stage == "Extended":
                share_state.reps_data['left_counter'] += 1
                share_state.reps_data['left_stage'] = "Flexed"
                # Reset feedback after successful rep
                share_state.form_feedback = ""
            
            # State machine for right arm
            if right_angle > EXTEND_THRESHOLD:
                if right_stage != "Extended":
                    share_state.reps_data['right_stage'] = "Extended"
            elif right_angle < FLEX_THRESHOLD and right_stage == "Extended":
                share_state.reps_data['right_counter'] += 1
                share_state.reps_data['right_stage'] = "Flexed"
                # Reset feedback after successful rep
                share_state.form_feedback = ""
            
            # Update total counter
            share_state.reps_data['counter'] = (
                share_state.reps_data['left_counter'] + share_state.reps_data['right_counter']
            )
            
            # Form feedback - check for swinging
            if left_angle < 30 and right_angle < 30:
                # Both arms are at top position, check if shoulders moved too much
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                
                # Calculate shoulder movement relative to hips
                shoulder_movement = abs(left_shoulder.x - left_hip.x) + abs(right_shoulder.x - right_hip.x)
                if shoulder_movement > 0.3:  # Too much swinging
                    share_state.form_feedback = "Don't swing your body"
                    if SOUND_ENABLED:
                        threading.Thread(target=play_wrong_sound, daemon=True).start()
                    
        except Exception as e:
            logger.error(f"Error in bicep curl processing: {e}")
    
    @staticmethod
    def squat(landmarks):
        """Squat processor - Biomechanically accurate"""
        try:
            # Get landmarks
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            # Check visibility
            if not all([check_visibility(left_hip), check_visibility(left_knee), check_visibility(left_ankle),
                       check_visibility(right_hip), check_visibility(right_knee), check_visibility(right_ankle)]):
                return
            
            # Prepare coordinates
            left_coords = [
                [left_hip.x, left_hip.y],
                [left_knee.x, left_knee.y],
                [left_ankle.x, left_ankle.y]
            ]
            
            right_coords = [
                [right_hip.x, right_hip.y],
                [right_knee.x, right_knee.y],
                [right_ankle.x, right_ankle.y]
            ]
            
            # Calculate knee angles
            left_angle = calculate_angle(left_coords[0], left_coords[1], left_coords[2])
            right_angle = calculate_angle(right_coords[0], right_coords[1], right_coords[2])
            
            # Calculate average angle
            avg_angle = (left_angle + right_angle) / 2
            
            # Biomechanical thresholds for squat
            STAND_THRESHOLD = 150  # Standing position
            SQUAT_THRESHOLD = 90   # Proper squat depth
            
            # Get current state
            current_stage = share_state.reps_data['stage']
            
            # State machine for squat
            if avg_angle > STAND_THRESHOLD:
                if current_stage != "Up":
                    share_state.reps_data['stage'] = "Up"
            elif avg_angle < SQUAT_THRESHOLD and current_stage == "Up":
                share_state.reps_data['counter'] += 1
                share_state.reps_data['stage'] = "Down"
                # Reset feedback after successful rep
                share_state.form_feedback = ""
            
            # Form feedback - check back angle
            # Calculate back angle (shoulder-hip-knee)
            left_back_angle = calculate_angle(
                [left_shoulder.x, left_shoulder.y],
                [left_hip.x, left_hip.y],
                [left_knee.x, left_knee.y]
            )
            
            right_back_angle = calculate_angle(
                [right_shoulder.x, right_shoulder.y],
                [right_hip.x, right_hip.y],
                [right_knee.x, right_knee.y]
            )
            
            avg_back_angle = (left_back_angle + right_back_angle) / 2
            
            if avg_back_angle < 80:  # Back is too hunched
                share_state.form_feedback = "Keep your back straight"
                if SOUND_ENABLED:
                    threading.Thread(target=play_wrong_sound, daemon=True).start()
                    
        except Exception as e:
            logger.error(f"Error in squat processing: {e}")
    
    @staticmethod
    def pushup(landmarks):
        """Pushup processor - Biomechanically accurate"""
        try:
            # Get landmarks
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            # Check visibility
            if not all([check_visibility(left_shoulder), check_visibility(left_elbow), check_visibility(left_wrist),
                       check_visibility(right_shoulder), check_visibility(right_elbow), check_visibility(right_wrist)]):
                return
            
            # Prepare coordinates for angle calculation
            left_elbow_angle = calculate_angle(
                [left_shoulder.x, left_shoulder.y],
                [left_elbow.x, left_elbow.y],
                [left_wrist.x, left_wrist.y]
            )
            
            left_shoulder_angle = calculate_angle(
                [left_hip.x, left_hip.y],
                [left_shoulder.x, left_shoulder.y],
                [left_elbow.x, left_elbow.y]
            )
            
            right_elbow_angle = calculate_angle(
                [right_shoulder.x, right_shoulder.y],
                [right_elbow.x, right_elbow.y],
                [right_wrist.x, right_wrist.y]
            )
            
            right_shoulder_angle = calculate_angle(
                [right_hip.x, right_hip.y],
                [right_shoulder.x, right_shoulder.y],
                [right_elbow.x, right_elbow.y]
            )
            
            # Biomechanical thresholds for pushup
            UP_THRESHOLD = 140     # Up position
            DOWN_THRESHOLD = 90    # Down position
            
            # Get current states
            left_stage = share_state.reps_data['left_stage']
            right_stage = share_state.reps_data['right_stage']
            
            # State machine for left arm
            avg_left_angle = (left_elbow_angle + left_shoulder_angle) / 2
            if avg_left_angle > UP_THRESHOLD:
                if left_stage != "Up":
                    share_state.reps_data['left_stage'] = "Up"
            elif avg_left_angle < DOWN_THRESHOLD and left_stage == "Up":
                share_state.reps_data['left_counter'] += 1
                share_state.reps_data['left_stage'] = "Down"
            
            # State machine for right arm
            avg_right_angle = (right_elbow_angle + right_shoulder_angle) / 2
            if avg_right_angle > UP_THRESHOLD:
                if right_stage != "Up":
                    share_state.reps_data['right_stage'] = "Up"
            elif avg_right_angle < DOWN_THRESHOLD and right_stage == "Up":
                share_state.reps_data['right_counter'] += 1
                share_state.reps_data['right_stage'] = "Down"
            
            # Update total counter
            share_state.reps_data['counter'] = (
                share_state.reps_data['left_counter'] + share_state.reps_data['right_counter']
            )
            
            # Form feedback - check for body sagging
            # Calculate hip height relative to shoulders
            left_hip_height = left_hip.y
            left_shoulder_height = left_shoulder.y
            hip_drop = left_hip_height - left_shoulder_height
            
            if hip_drop > 0.1:  # Hips are sagging too low
                share_state.form_feedback = "Keep your body straight"
                if SOUND_ENABLED:
                    threading.Thread(target=play_wrong_sound, daemon=True).start()
                    
        except Exception as e:
            logger.error(f"Error in pushup processing: {e}")
    
    @staticmethod
    def shoulder_press(landmarks):
        """Shoulder press processor - Biomechanically accurate"""
        try:
            # Get landmarks
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            # Check visibility
            if not all([check_visibility(left_shoulder), check_visibility(left_elbow), check_visibility(left_wrist),
                       check_visibility(right_shoulder), check_visibility(right_elbow), check_visibility(right_wrist)]):
                return
            
            # Prepare coordinates
            left_coords = [
                [left_shoulder.x, left_shoulder.y],
                [left_elbow.x, left_elbow.y],
                [left_wrist.x, left_wrist.y]
            ]
            
            right_coords = [
                [right_shoulder.x, right_shoulder.y],
                [right_elbow.x, right_elbow.y],
                [right_wrist.x, right_wrist.y]
            ]
            
            # Calculate angles
            left_angle = calculate_angle(left_coords[0], left_coords[1], left_coords[2])
            right_angle = calculate_angle(right_coords[0], right_coords[1], right_coords[2])
            
            # Calculate average angle
            avg_angle = (left_angle + right_angle) / 2
            
            # Biomechanical thresholds for shoulder press
            DOWN_THRESHOLD = 60    # Arms down
            UP_THRESHOLD = 150     # Arms up (overhead)
            
            # Get current state
            current_stage = share_state.reps_data['stage']
            
            # State machine for shoulder press
            if avg_angle < DOWN_THRESHOLD:
                if current_stage != "Down":
                    share_state.reps_data['stage'] = "Down"
            elif avg_angle > UP_THRESHOLD and current_stage == "Down":
                share_state.reps_data['counter'] += 1
                share_state.reps_data['stage'] = "Up"
                # Reset feedback after successful rep
                share_state.form_feedback = ""
            
            # Form feedback - check elbow flare
            # Calculate elbow position relative to shoulder
            left_elbow_position = abs(left_elbow.x - left_shoulder.x)
            right_elbow_position = abs(right_elbow.x - right_shoulder.x)
            
            if left_elbow_position > 0.2 or right_elbow_position > 0.2:
                share_state.form_feedback = "Keep elbows close to body"
                if SOUND_ENABLED:
                    threading.Thread(target=play_wrong_sound, daemon=True).start()
                    
        except Exception as e:
            logger.error(f"Error in shoulder press processing: {e}")
    
    @staticmethod
    def lateral_raise(landmarks):
        """Lateral raise processor - Biomechanically accurate"""
        try:
            # Get landmarks
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            # Check visibility
            if not all([check_visibility(left_shoulder), check_visibility(left_elbow), check_visibility(left_wrist),
                       check_visibility(right_shoulder), check_visibility(right_elbow), check_visibility(right_wrist)]):
                return
            
            # Calculate angles (shoulder abduction angle)
            left_angle = calculate_angle(
                [left_hip.x, left_hip.y],
                [left_shoulder.x, left_shoulder.y],
                [left_elbow.x, left_elbow.y]
            )
            
            right_angle = calculate_angle(
                [right_hip.x, right_hip.y],
                [right_shoulder.x, right_shoulder.y],
                [right_elbow.x, right_elbow.y]
            )
            
            # Calculate average angle
            avg_angle = (left_angle + right_angle) / 2
            
            # Biomechanical thresholds for lateral raise
            DOWN_THRESHOLD = 20    # Arms down
            UP_THRESHOLD = 90      # Arms parallel to floor
            
            # Get current state
            current_stage = share_state.reps_data['stage']
            
            # State machine for lateral raise
            if avg_angle < DOWN_THRESHOLD:
                if current_stage != "Down":
                    share_state.reps_data['stage'] = "Down"
            elif avg_angle > UP_THRESHOLD and current_stage == "Down":
                share_state.reps_data['counter'] += 1
                share_state.reps_data['stage'] = "Up"
                # Reset feedback after successful rep
                share_state.form_feedback = ""
            
            # Form feedback - check for elbow bending
            left_elbow_angle = calculate_angle(
                [left_shoulder.x, left_shoulder.y],
                [left_elbow.x, left_elbow.y],
                [left_wrist.x, left_wrist.y]
            )
            
            right_elbow_angle = calculate_angle(
                [right_shoulder.x, right_shoulder.y],
                [right_elbow.x, right_elbow.y],
                [right_wrist.x, right_wrist.y]
            )
            
            if left_elbow_angle < 160 or right_elbow_angle < 160:
                share_state.form_feedback = "Keep arms straight"
                if SOUND_ENABLED:
                    threading.Thread(target=play_wrong_sound, daemon=True).start()
                    
        except Exception as e:
            logger.error(f"Error in lateral raise processing: {e}")
    
    @staticmethod
    def dumbbell_row(landmarks):
        """Dumbbell row processor - Biomechanically accurate"""
        try:
            # Get landmarks
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            # Check visibility
            if not all([check_visibility(left_shoulder), check_visibility(left_elbow), check_visibility(left_wrist),
                       check_visibility(right_shoulder), check_visibility(right_elbow), check_visibility(right_wrist)]):
                return
            
            # Calculate elbow angles (back angle)
            left_angle = calculate_angle(
                [left_hip.x, left_hip.y],
                [left_shoulder.x, left_shoulder.y],
                [left_elbow.x, left_elbow.y]
            )
            
            right_angle = calculate_angle(
                [right_hip.x, right_hip.y],
                [right_shoulder.x, right_shoulder.y],
                [right_elbow.x, right_elbow.y]
            )
            
            # Biomechanical thresholds for dumbbell row
            DOWN_THRESHOLD = 50    # Arms extended
            UP_THRESHOLD = 90      # Elbows behind body
            
            # Get current states
            left_stage = share_state.reps_data['left_stage']
            right_stage = share_state.reps_data['right_stage']
            
            # State machine for left arm
            if left_angle < DOWN_THRESHOLD:
                if left_stage != "Down":
                    share_state.reps_data['left_stage'] = "Down"
            elif left_angle > UP_THRESHOLD and left_stage == "Down":
                share_state.reps_data['left_counter'] += 1
                share_state.reps_data['left_stage'] = "Up"
            
            # State machine for right arm
            if right_angle < DOWN_THRESHOLD:
                if right_stage != "Down":
                    share_state.reps_data['right_stage'] = "Down"
            elif right_angle > UP_THRESHOLD and right_stage == "Down":
                share_state.reps_data['right_counter'] += 1
                share_state.reps_data['right_stage'] = "Up"
            
            # Update total counter
            share_state.reps_data['counter'] = (
                share_state.reps_data['left_counter'] + share_state.reps_data['right_counter']
            )
            
            # Form feedback - check back angle
            left_shoulder_angle = calculate_angle(
                [left_hip.x, left_hip.y],
                [left_shoulder.x, left_shoulder.y],
                [left_elbow.x, left_elbow.y]
            )
            
            right_shoulder_angle = calculate_angle(
                [right_hip.x, right_hip.y],
                [right_shoulder.x, right_shoulder.y],
                [right_elbow.x, right_elbow.y]
            )
            
            if left_shoulder_angle < 70 or right_shoulder_angle < 70:
                share_state.form_feedback = "Keep back straight, don't hunch"
                if SOUND_ENABLED:
                    threading.Thread(target=play_wrong_sound, daemon=True).start()
                    
        except Exception as e:
            logger.error(f"Error in dumbbell row processing: {e}")

# ============ GLOBAL STATE MANAGEMENT ============

class GlobalState:
    """Thread-safe global state management"""
    def __init__(self):
        self.tracking_enabled = False
        self.selected_exercise = "bicep_curl"
        self.camera_active = True  # CHANGED: Camera starts ENABLED (matches frontend)
        self.camera_initialized = False
        self.camera_lock = threading.Lock()
        self.current_cap = None
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.last_frame_time = 0
        self.form_feedback = ""
        self.mediapipe_running = False
        self.start_time = None
        self.exercise_duration = 0
        self.workout_history = []
        self.streak_data = defaultdict(int)
        
        # Rep counters - use regular dict without lock for simplicity
        self.reps_data = {
            'counter': 0,
            'left_counter': 0,
            'right_counter': 0,
            'stage': 'Ready',
            'left_stage': 'Ready',
            'right_stage': 'Ready'
        }
        
        # Exercise processors
        self.exercise_processors = {
            'bicep_curl': ExerciseProcessors.bicep_curl,
            'squat': ExerciseProcessors.squat,
            'pushup': ExerciseProcessors.pushup,
            'shoulder_press': ExerciseProcessors.shoulder_press,
            'lateral_raise': ExerciseProcessors.lateral_raise,
            'dumbbell_row': ExerciseProcessors.dumbbell_row
        }
        
        self.calorie_factors = {
            'bicep_curl': 0.02,
            'squat': 0.05,
            'pushup': 0.08,
            'shoulder_press': 0.03,
            'lateral_raise': 0.025,
            'dumbbell_row': 0.04
        }
    
    def reset_reps(self):
        """Reset all rep counters"""
        self.reps_data = {
            'counter': 0,
            'left_counter': 0,
            'right_counter': 0,
            'stage': 'Ready',
            'left_stage': 'Ready',
            'right_stage': 'Ready'
        }
        self.form_feedback = ""
        self.start_time = None
        self.exercise_duration = 0
    
    def get_calories_burned(self, reps):
        """Calculate calories burned based on exercise type"""
        factor = self.calorie_factors.get(self.selected_exercise, 0.02)
        return round(reps * factor, 2)
    
    def get_duration(self):
        """Get exercise duration in seconds"""
        if self.start_time:
            return int(time.time() - self.start_time)
        return 0

# Initialize global state
share_state = GlobalState()

# ============ DATABASE MANAGEMENT ============

DATABASE = 'fitness_tracker.db'

@contextmanager
def get_db_connection():
    """Get a database connection with proper context management"""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def init_db():
    """Initialize the database with proper connection handling"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            # Create users table
            c.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create workouts table with calories
            c.execute('''
                CREATE TABLE IF NOT EXISTS workouts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    exercise_type TEXT NOT NULL,
                    reps INTEGER NOT NULL,
                    duration INTEGER DEFAULT 60,
                    calories_estimate REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')
            
            # Create streaks table
            c.execute('''
                CREATE TABLE IF NOT EXISTS streaks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    date DATE UNIQUE,
                    workouts_count INTEGER DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            ''')
            
            # Create index for faster queries
            c.execute('CREATE INDEX IF NOT EXISTS idx_workouts_user_id ON workouts(user_id)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_workouts_created_at ON workouts(created_at)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_streaks_user_date ON streaks(user_id, date)')
            
            conn.commit()
            logger.info("✅ Database initialized successfully!")
            
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")
        raise

# ============ CAMERA MANAGEMENT ============

class CameraManager:
    """Thread-safe camera management"""
    
    @staticmethod
    def init_camera():
        """Initialize camera in background"""
        with share_state.camera_lock:
            if share_state.current_cap is not None and share_state.current_cap.isOpened():
                return share_state.current_cap
            
            try:
                logger.info("📷 Initializing camera...")
                
                # Release any existing camera
                if share_state.current_cap:
                    try:
                        share_state.current_cap.release()
                    except:
                        pass
                
                # Try different camera indices
                for camera_index in [0, 1]:
                    try:
                        cap = cv2.VideoCapture(camera_index)
                        if not cap.isOpened():
                            continue
                        
                        # Set optimized camera settings
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                        
                        # Test frame
                        ret, frame = cap.read()
                        if ret:
                            logger.info(f"✅ Camera {camera_index}: Initialized successfully")
                            share_state.current_cap = cap
                            share_state.camera_initialized = True
                            return cap
                        else:
                            cap.release()
                    except Exception as e:
                        logger.warning(f"Camera {camera_index} failed: {e}")
                        continue
                
                logger.warning("⚠️ No working camera found")
                share_state.camera_initialized = False
                return None
                
            except Exception as e:
                logger.error(f"❌ Camera initialization error: {e}")
                share_state.camera_initialized = False
                return None
    
    @staticmethod
    def release_camera():
        """Release camera resources"""
        with share_state.camera_lock:
            if share_state.current_cap is not None:
                try:
                    share_state.current_cap.release()
                    logger.info("📷 Camera released")
                except Exception as e:
                    logger.error(f"❌ Error releasing camera: {e}")
                finally:
                    share_state.current_cap = None
                    share_state.camera_initialized = False
    
    @staticmethod
    def get_frame():
        """Get a single frame from camera"""
        if not share_state.camera_active or share_state.current_cap is None:
            return None
        
        try:
            ret, frame = share_state.current_cap.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Mirror effect
                return frame
        except Exception as e:
            logger.error(f"❌ Error reading frame: {e}")
        
        return None

# ============ VIDEO PROCESSING ============

# Initialize MediaPipe once for the entire application
pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1,
    smooth_landmarks=True,
    smooth_segmentation=True
)

def process_frame_with_mediapipe(frame):
    """Process a frame with MediaPipe"""
    if frame is None:
        return None, None
    
    try:
        # Resize for display
        display_frame = cv2.resize(frame, (640, 480))
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = pose.process(rgb_frame)
        
        # Only draw if tracking is enabled AND landmarks detected
        if share_state.tracking_enabled and results.pose_landmarks:
            # Draw landmarks with stable colors
            mp_drawing.draw_landmarks(
                display_frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(247, 117, 66), thickness=3, circle_radius=3),
                mp_drawing.DrawingSpec(color=(247, 66, 230), thickness=3, circle_radius=3)
            )
            
            # Process exercise
            processor = share_state.exercise_processors.get(share_state.selected_exercise)
            if processor:
                processor(results.pose_landmarks.landmark)
        
        return display_frame, results
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return cv2.resize(frame, (640, 480)), None

# ============ STREAK CALCULATION ============

def calculate_streak(user_id):
    """Calculate current streak for a user"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            # Get all workout dates for user
            c.execute('''
                SELECT DATE(created_at) as workout_date 
                FROM workouts 
                WHERE user_id = ?
                GROUP BY DATE(created_at)
                ORDER BY workout_date DESC
                LIMIT 30
            ''', (user_id,))
            
            dates = [row['workout_date'] for row in c.fetchall()]
            
            if not dates:
                return 0
            
            # Calculate streak
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            
            streak = 0
            current_date = today
            
            # Check if user worked out today
            if dates[0] == today.isoformat():
                streak = 1
                # Check consecutive days backwards
                for i in range(len(dates)):
                    workout_date = datetime.strptime(dates[i], '%Y-%m-%d').date()
                    if workout_date == current_date:
                        if i > 0:
                            # Check if consecutive
                            prev_workout_date = datetime.strptime(dates[i-1], '%Y-%m-%d').date()
                            if prev_workout_date == current_date - timedelta(days=1):
                                streak += 1
                                current_date = prev_workout_date
                            else:
                                break
                    else:
                        break
            # Check if user worked out yesterday
            elif dates[0] == yesterday.isoformat():
                streak = 1
                # Check consecutive days backwards
                current_date = yesterday
                for i in range(len(dates)):
                    workout_date = datetime.strptime(dates[i], '%Y-%m-%d').date()
                    if workout_date == current_date:
                        if i > 0:
                            prev_workout_date = datetime.strptime(dates[i-1], '%Y-%m-%d').date()
                            if prev_workout_date == current_date - timedelta(days=1):
                                streak += 1
                                current_date = prev_workout_date
                            else:
                                break
                    else:
                        break
            
            return streak
            
    except Exception as e:
        logger.error(f"❌ Error calculating streak: {e}")
        return 0

def update_streak(user_id):
    """Update streak for today"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            today = datetime.now().date().isoformat()
            
            c.execute('''
                INSERT OR REPLACE INTO streaks (user_id, date, workouts_count)
                VALUES (?, ?, COALESCE(
                    (SELECT workouts_count + 1 FROM streaks WHERE user_id = ? AND date = ?),
                    1
                ))
            ''', (user_id, today, user_id, today))
            
            conn.commit()
    except Exception as e:
        logger.error(f"❌ Error updating streak: {e}")

# ============ VIDEO FRAME GENERATOR ============

def generate_frame():
    """Generate video frames - Camera feed only when enabled"""
    logger.info("🎥 Starting video feed generator")
    
    last_processing_time = time.time()
    processing_interval = 0.033  # Process every 33ms (30 FPS)
    
    try:
        while True:
            current_time = time.time()
            
            # Check if camera should be active
            if not share_state.camera_active:
                # Return black frame when camera is disabled
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(0.1)
                continue
            
            # Get frame from camera
            frame = CameraManager.get_frame()
            if frame is None:
                # Camera error frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera ERROR", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(0.1)
                continue
            
            # Process frame consistently
            if current_time - last_processing_time >= processing_interval:
                processed_frame, _ = process_frame_with_mediapipe(frame)
                last_processing_time = current_time
            else:
                # Use previous processed frame or just resize
                processed_frame = cv2.resize(frame, (640, 480))
            
            # Encode frame
            _, jpeg = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Yield frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
    except GeneratorExit:
        logger.info("🎥 Video feed client disconnected")
    except Exception as e:
        logger.error(f"Error in video feed: {e}")
        try:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Stream ERROR", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        except:
            pass

# ============ FLASK APP CONFIGURATION ============

# Initialize database
init_db()

# Initialize camera on startup (since camera starts enabled)
CameraManager.init_camera()

# Add CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# ============ AUTHENTICATION HELPERS ============

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('index'))  # Changed from landing to index
        return f(*args, **kwargs)
    return decorated_function

def check_session_auth():
    return 'user_id' in session

# ============ MAIN ROUTES ============

@app.route('/')
def index():
    if check_session_auth():
        return redirect(url_for('dashboard'))
    else:
        # Serve landing page directly at /
        return render_template('landing.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session.get('user_id')
    username = session.get('username')
    
    user_data = {
        'id': user_id,
        'username': username,
        'name': session.get('name'),
        'email': session.get('email')
    }
    
    return render_template('index.html', user_data=user_data)

@app.route('/signup')
def signup():
    if check_session_auth():
        return redirect(url_for('dashboard'))
    return render_template('auth.html')

@app.route('/login')
def login():
    if check_session_auth():
        return redirect(url_for('dashboard'))
    return render_template('auth.html')

# ============ AUTHENTICATION API ROUTES ============

@app.route('/api/auth/signup', methods=['POST'])
def api_signup():
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        
        if not all([name, email, password]):
            return jsonify({'error': 'All fields are required'}), 400
        
        username = email.split('@')[0]
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        with get_db_connection() as conn:
            c = conn.cursor()
            
            c.execute('SELECT id FROM users WHERE email = ?', (email,))
            if c.fetchone():
                return jsonify({'error': 'Email already exists'}), 400
            
            c.execute('INSERT INTO users (name, username, email, password_hash) VALUES (?, ?, ?, ?)',
                     (name, username, email, password_hash))
            conn.commit()
            user_id = c.lastrowid
            
            c.execute('SELECT id, name, username, email FROM users WHERE id = ?', (user_id,))
            user = c.fetchone()
            
            session.permanent = True
            session['user_id'] = user_id
            session['username'] = username
            session['email'] = email
            session['name'] = name
            
            return jsonify({
                'message': 'Signup successful',
                'user': dict(user),
                'redirect': '/dashboard'
            }), 201
            
    except Exception as e:
        logger.error(f"❌ Signup error: {e}")
        return jsonify({'error': 'Server error'}), 500

@app.route('/api/auth/signin', methods=['POST'])
def api_signin():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not all([email, password]):
            return jsonify({'error': 'Email and password are required'}), 400
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT id, name, username, email FROM users WHERE email = ? AND password_hash = ?',
                     (email, password_hash))
            user = c.fetchone()
            
            if user:
                session.permanent = True
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['email'] = user['email']
                session['name'] = user['name']
                
                return jsonify({
                    'message': 'Login successful',
                    'user': dict(user),
                    'redirect': '/dashboard'
                })
            else:
                return jsonify({'error': 'Invalid email or password'}), 401
            
    except Exception as e:
        logger.error(f"❌ Signin error: {e}")
        return jsonify({'error': 'Server error'}), 500

@app.route('/api/auth/logout', methods=['POST'])
@login_required
def api_logout():
    session.clear()
    share_state.camera_active = True  # Reset to enabled for next session
    CameraManager.release_camera()
    return jsonify({'message': 'Logout successful', 'redirect': '/'})

@app.route('/api/auth/check', methods=['GET'])
def check_auth():
    if check_session_auth():
        return jsonify({
            'authenticated': True,
            'user': {
                'id': session.get('user_id'),
                'name': session.get('name'),
                'username': session.get('username'),
                'email': session.get('email')
            }
        })
    else:
        return jsonify({'authenticated': False})

# ============ USER STATISTICS API ============

@app.route('/api/user/stats', methods=['GET'])
@login_required
def get_user_stats():
    user_id = session.get('user_id')
    
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            c.execute('SELECT COUNT(*) as count FROM workouts WHERE user_id = ?', (user_id,))
            total_workouts = c.fetchone()['count']
            
            c.execute('SELECT COALESCE(SUM(reps), 0) as total FROM workouts WHERE user_id = ?', (user_id,))
            total_reps = c.fetchone()['total']
            
            c.execute('SELECT COALESCE(SUM(calories_estimate), 0) as total_calories FROM workouts WHERE user_id = ?', (user_id,))
            total_calories = c.fetchone()['total_calories']
            
            # Calculate streak
            streak = calculate_streak(user_id)
            
            return jsonify({
                'total_workouts': total_workouts,
                'total_reps': total_reps,
                'total_calories': round(float(total_calories), 1),
                'current_streak': streak
            })
    except Exception as e:
        logger.error(f"❌ Error fetching user stats: {e}")
        return jsonify({
            'total_workouts': 0,
            'total_reps': 0,
            'total_calories': 0,
            'current_streak': 0
        })

@app.route('/api/workouts', methods=['GET'])
@login_required
def get_workouts():
    try:
        user_id = session.get('user_id')
        
        with get_db_connection() as conn:
            c = conn.cursor()
            
            c.execute('''
                SELECT exercise_type as exercise, reps, duration, calories_estimate as calories, created_at as date
                FROM workouts 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT 10
            ''', (user_id,))
            
            workouts = [dict(row) for row in c.fetchall()]
            
            return jsonify({'workouts': workouts})
    except Exception as e:
        logger.error(f"❌ Error fetching workouts: {e}")
        return jsonify({'workouts': []})

# ============ VIDEO AND EXERCISE ROUTES ============

@app.route('/video_feed')
def video_feed():
    """Video feed endpoint with proper headers"""
    return Response(generate_frame(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame',
                   headers={
                       'Cache-Control': 'no-cache, no-store, must-revalidate',
                       'Pragma': 'no-cache',
                       'Expires': '0'
                   })

@app.route('/enable_camera', methods=['POST'])
@login_required
def enable_camera():
    try:
        # Initialize camera if not already done
        CameraManager.init_camera()
        share_state.camera_active = True
        logger.info("📷 Camera enabled")
        return jsonify({'success': True, 'message': 'Camera enabled'})
    except Exception as e:
        logger.error(f"❌ Error enabling camera: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/disable_camera', methods=['POST'])
@login_required
def disable_camera():
    try:
        share_state.camera_active = False
        logger.info("📷 Camera disabled")
        return jsonify({'success': True, 'message': 'Camera disabled'})
    except Exception as e:
        logger.error(f"❌ Error disabling camera: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/select_exercise', methods=['POST'])
@login_required
def select_exercise():
    try:
        data = request.get_json() if request.is_json else request.form
        new_exercise = data.get('exercise', 'bicep_curl')
        
        if new_exercise not in share_state.exercise_processors:
            return jsonify({'error': 'Invalid exercise'}), 400
        
        share_state.tracking_enabled = False
        share_state.reset_reps()
        share_state.selected_exercise = new_exercise
        
        logger.info(f"🔄 Selected exercise: {new_exercise}")
        return jsonify({
            'success': True,
            'message': f'Exercise switched to {new_exercise}',
            'exercise': new_exercise
        }), 200
            
    except Exception as e:
        logger.error(f"❌ Error selecting exercise: {e}")
        return jsonify({'success': False, 'error': 'Error selecting exercise'}), 500

@app.route('/get_reps')
@login_required
def get_reps():
    """Get current reps data"""
    duration = share_state.get_duration()
    
    return jsonify({
        'counter': share_state.reps_data['counter'],
        'reps': share_state.reps_data['counter'],
        'stage': share_state.reps_data['stage'],
        'left_reps': share_state.reps_data['left_counter'],
        'left_stage': share_state.reps_data['left_stage'],
        'right_reps': share_state.reps_data['right_counter'],
        'right_stage': share_state.reps_data['right_stage'],
        'tracking': share_state.tracking_enabled,
        'exercise': share_state.selected_exercise,
        'feedback': share_state.form_feedback or "",
        'duration': duration  # Add duration to response
    })

@app.route('/start_tracking', methods=['POST'])
@login_required
def start_tracking():
    try:
        share_state.tracking_enabled = True
        share_state.reset_reps()
        share_state.start_time = time.time()  # Start timer
        logger.info(f"▶️ Tracking started for {share_state.selected_exercise}")
        return jsonify({
            'success': True,
            'message': 'Tracking started',
            'exercise': share_state.selected_exercise,
            'start_time': share_state.start_time
        }), 200
    except Exception as e:
        logger.error(f"❌ Error starting tracking: {e}")
        return jsonify({'success': False, 'error': 'Failed to start tracking'}), 500

@app.route('/stop_tracking', methods=['POST'])
@login_required
def stop_tracking():
    try:
        share_state.tracking_enabled = False
        logger.info(f"⏹️ Tracking stopped for {share_state.selected_exercise}")
        
        # Calculate duration
        duration = share_state.get_duration()
        
        # Save workout
        user_id = session.get('user_id')
        if user_id and share_state.reps_data['counter'] > 0:
            try:
                total_reps = share_state.reps_data['counter']
                calories_estimate = share_state.get_calories_burned(total_reps)
                
                with get_db_connection() as conn:
                    c = conn.cursor()
                    c.execute('''
                        INSERT INTO workouts (user_id, exercise_type, reps, duration, calories_estimate) 
                        VALUES (?, ?, ?, ?, ?)
                    ''', (user_id, share_state.selected_exercise, total_reps, duration, calories_estimate))
                    conn.commit()
                
                # Update streak
                update_streak(user_id)
                
                logger.info(f"💾 Saved workout: {total_reps} reps, {calories_estimate} calories, {duration}s duration")
            except Exception as e:
                logger.error(f"❌ Error saving workout: {e}")
        
        share_state.start_time = None  # Reset timer
        
        return jsonify({
            'success': True,
            'message': 'Tracking stopped',
            'final_reps': share_state.reps_data['counter'],
            'duration': duration
        }), 200
    except Exception as e:
        logger.error(f"❌ Error stopping tracking: {e}")
        return jsonify({'success': False, 'error': 'Failed to stop tracking'}), 500

@app.route('/reset_reps', methods=['POST'])
@login_required
def reset_reps():
    try:
        share_state.reset_reps()
        logger.info("🔄 Counters reset")
        return jsonify({'success': True, 'message': 'Counters reset'}), 200
    except Exception as e:
        logger.error(f"❌ Error resetting reps: {e}")
        return jsonify({'success': False, 'error': 'Failed to reset counters'}), 500

# ============ MISC ROUTES ============

@app.route('/api/exercises', methods=['GET'])
@login_required
def get_exercises():
    exercises = [
        {'id': 'bicep_curl', 'name': 'Bicep Curl'},
        {'id': 'squat', 'name': 'Squat'},
        {'id': 'pushup', 'name': 'Push Up'},
        {'id': 'shoulder_press', 'name': 'Shoulder Press'},
        {'id': 'lateral_raise', 'name': 'Lateral Raise'},
        {'id': 'dumbbell_row', 'name': 'Dumbbell Row'}
    ]
    return jsonify({'exercises': exercises})

# ============ CLEANUP ============

@atexit.register
def cleanup():
    """Cleanup resources on shutdown"""
    logger.info("🛑 Cleaning up resources...")
    CameraManager.release_camera()
    # Close MediaPipe
    pose.close()

# ============ STARTUP ============

if __name__ == "__main__":
    print("🚀 AI Fitness Trainer Started - FIXED FOR FRONTEND!")
    print("=" * 50)
    print(f"📊 Loaded {len(share_state.exercise_processors)} exercises")
    print("🔊 Sound enabled:", SOUND_ENABLED)
    print("🎥 VIDEO SYSTEM CONFIGURATION:")
    print("   • Camera starts ENABLED (matches frontend expectation)")
    print("   • Shows live camera feed on page load")
    print("   • Click 'Disable Camera' to show demo GIF overlay")
    print("   • Click 'Enable Camera' to go back to live feed")
    print("📊 Stats FIXES:")
    print("   • Day streak calculation working")
    print("   • Exercise duration tracking")
    print("🌐 Landing Page: http://localhost:8080")
    print("📄 Terms & Conditions: http://localhost:8080/terms")
    print("🔒 Privacy Policy: http://localhost:8080/privacy")
    print("🏋️‍♂️ Dashboard: http://localhost:8080/dashboard")
    print("🔐 Authentication: http://localhost:8080/signup")
    print("🎥 Video Feed: http://localhost:8080/video_feed")
    print("=" * 50)
    print("⚡ Starting server on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True, use_reloader=False)