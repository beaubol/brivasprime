import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import random
from typing import List, Optional

# --- CONFIGURATION & CONSTANTS ---
# Thresholds for "Motion Proof" Liveness
EAR_THRESHOLD = 0.21    # Eye Aspect Ratio < 0.21 implies closed eyes
MAR_THRESHOLD = 0.4     # Mouth Aspect Ratio > 0.4 implies smile/open
YAW_THRESHOLD = 25      # Degrees to turn head left/right
PITCH_THRESHOLD = 15    # Degrees to look up/down
FACE_CLOSE_RATIO = 0.45 # Face must occupy 45% of screen for "Move Closer"

# --- INITIALIZE NEURAL NETWORKS ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.8
)

app = FastAPI(title="Biometric Gateway API", version="2.0.0")

# Allow CORS for mobile apps to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA STRUCTURES ---
class ChallengeResponse(BaseModel):
    challenge_id: str
    instruction: str
    icon: str

class VerificationResult(BaseModel):
    verified: bool
    message: str
    confidence: float

# --- UTILITY FUNCTIONS ---

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    """Calculates Eye Aspect Ratio (EAR) to detect blinking."""
    # Indices: [left, right, top, bottom] - Simplified for brevity
    # Actual MediaPipe indices need to be mapped correctly
    # Using vertical distance / horizontal distance
    try:
        # Convert normalized landmarks to pixels
        coords = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_indices]
        
        # Vertical lines
        v1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
        v2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
        # Horizontal line
        h = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
        
        return (v1 + v2) / (2.0 * h)
    except:
        return 0.0

def get_head_pose(landmarks, img_w, img_h):
    """Estimates Head Pose (Yaw, Pitch, Roll) using solvePnP."""
    # 3D Model points (generic face)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right Mouth corner
    ])

    # 2D Image points from MediaPipe
    image_points = np.array([
        (landmarks[1].x * img_w, landmarks[1].y * img_h),     # Nose tip
        (landmarks[152].x * img_w, landmarks[152].y * img_h), # Chin
        (landmarks[263].x * img_w, landmarks[263].y * img_h), # Left eye
        (landmarks[33].x * img_w, landmarks[33].y * img_h),   # Right eye
        (landmarks[61].x * img_w, landmarks[61].y * img_h),   # Left mouth
        (landmarks[291].x * img_w, landmarks[291].y * img_h)  # Right mouth
    ], dtype="double")

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    # Convert rotation vector to Euler angles
    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    # angles[0]=pitch, angles[1]=yaw, angles[2]=roll
    return angles[0] * 360, angles[1] * 360, angles[2] * 360

# --- API ENDPOINTS ---

@app.get("/challenge/new", response_model=ChallengeResponse)
async def get_challenge():
    """Generates a random liveness challenge."""
    challenges = [
        {"id": "blink", "text": "Blink your eyes", "icon": "👁️"},
        {"id": "smile", "text": "Smile widely", "icon": "😊"},
        {"id": "turn_left", "text": "Turn head left", "icon": "⬅️"},
        {"id": "turn_right", "text": "Turn head right", "icon": "➡️"},
        {"id": "close", "text": "Move closer to camera", "icon": "🔍"}
    ]
    choice = random.choice(challenges)
    return ChallengeResponse(
        challenge_id=choice['id'],
        instruction=choice['text'],
        icon=choice['icon']
    )

@app.post("/verify/liveness", response_model=VerificationResult)
async def verify_liveness(challenge_id: str, file: UploadFile = File(...)):
    """
    Analyzes the uploaded frame to verify if the user performed the action.
    """
    # 1. Read Image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w, _ = frame.shape

    # 2. Get Mesh
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return VerificationResult(verified=False, message="No face detected", confidence=0.0)

    landmarks = results.multi_face_landmarks[0].landmark
    verified = False
    confidence = 0.0
    message = "Action failed"

    # 3. Analyze based on Challenge ID
    if challenge_id == "blink":
        # Calculate EAR (Indices for Left/Right eyes)
        left_eye = [362, 385, 387, 263, 373, 380]
        right_eye = [33, 160, 158, 133, 153, 144]
        ear_left = calculate_ear(landmarks, left_eye, w, h)
        ear_right = calculate_ear(landmarks, right_eye, w, h)
        avg_ear = (ear_left + ear_right) / 2.0
        
        if avg_ear < EAR_THRESHOLD:
            verified = True
            message = "Blink detected"
            confidence = 0.99

    elif challenge_id == "smile":
        # Simple MAR calculation or check lip corners
        # Indices: 61 (left corner), 291 (right corner), 0 (top lip), 17 (bottom lip)
        mouth_w = abs(landmarks[61].x - landmarks[291].x)
        mouth_h = abs(landmarks[0].y - landmarks[17].y)
        ratio = mouth_h / mouth_w if mouth_w > 0 else 0
        
        if ratio > MAR_THRESHOLD: # Simple heuristic
            verified = True
            message = "Smile detected"
            confidence = 0.95

    elif challenge_id in ["turn_left", "turn_right"]:
        pitch, yaw, roll = get_head_pose(landmarks, w, h)
        
        if challenge_id == "turn_left" and yaw < -YAW_THRESHOLD:
            verified = True
            message = "Left turn detected"
            confidence = abs(yaw) / 90.0
        elif challenge_id == "turn_right" and yaw > YAW_THRESHOLD:
            verified = True
            message = "Right turn detected"
            confidence = abs(yaw) / 90.0

    elif challenge_id == "close":
        # Check bounding box area relative to frame
        # Simple approx: nose to side of face distance
        face_width_ratio = abs(landmarks[234].x - landmarks[454].x)
        if face_width_ratio > FACE_CLOSE_RATIO:
            verified = True
            message = "Proximity verified"
            confidence = face_width_ratio

    # 4. Return Result
    # NOTE: For "Authentication" (Identity), this is where you would 
    # pass `frame` to a library like DeepFace.verify(frame, db_path="gigantic_db/")
    
    return VerificationResult(
        verified=verified,
        message=message,
        confidence=confidence
    )
