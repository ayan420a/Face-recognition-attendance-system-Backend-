import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import List, Optional

import cv2
import dlib
import face_recognition
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from scipy.spatial import distance as dist
import jwt

# ------------ CONFIG -----------------
IMAGE_DIR = "photos"
EXCEL_FILE = "attendance.xlsx"
LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat"
USERS_FILE = "users.json"
JWT_SECRET = os.environ.get("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

os.makedirs(IMAGE_DIR, exist_ok=True)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()


# ------------ AUTH HELPERS -----------

def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def _save_users(users: dict) -> None:
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _create_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    try:
        payload = jwt.decode(
            credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM]
        )
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


import re

def _get_safe_username(username: str) -> str:
    # Replace any character that is not alphanumeric, dot, or hyphen with underscore
    return re.sub(r'[^a-zA-Z0-9.-]', '_', username.lower())


# ------------ AUTH MODELS ------------

class SignupRequest(BaseModel):
    username: str
    password: str
    fullName: str


class LoginRequest(BaseModel):
    username: str
    password: str


# ------------ AUTH ROUTES ------------

@app.get("/")
def home():
    return {"message": "Backend running"}


@app.post("/api/auth/signup")
def signup(req: SignupRequest):
    users = _load_users()
    if req.username.lower() in users:
        raise HTTPException(status_code=400, detail="Username already exists")

    users[req.username.lower()] = {
        "password": _hash_password(req.password),
        "fullName": req.fullName,
        "created": datetime.now().isoformat(),
    }
    _save_users(users)
    token = _create_token(req.username.lower())
    return {
        "token": token,
        "username": req.username.lower(),
        "fullName": req.fullName,
    }


@app.post("/api/auth/login")
def login(req: LoginRequest):
    users = _load_users()
    user = users.get(req.username.lower())
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if user["password"] != _hash_password(req.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = _create_token(req.username.lower())
    return {
        "token": token,
        "username": req.username.lower(),
        "fullName": user["fullName"],
    }


@app.get("/api/auth/me")
def get_me(username: str = Depends(_verify_token)):
    users = _load_users()
    user = users.get(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"username": username, "fullName": user["fullName"]}


# ------------ FACE DATA -------------

def load_known_faces_for_user(username: str) -> tuple[List[np.ndarray], List[str]]:
    safe_user = _get_safe_username(username)
    user_dir = os.path.join(IMAGE_DIR, safe_user)
    os.makedirs(user_dir, exist_ok=True)
    
    encodings = []
    names = []
    
    for filename in os.listdir(user_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(user_dir, filename)
            name = os.path.splitext(filename)[0]
            try:
                img = face_recognition.load_image_file(path)
                file_encodings = face_recognition.face_encodings(img)
                if not file_encodings:
                    print(f"[WARN] No face found in {filename} for user {username}")
                    continue
                encodings.append(file_encodings[0])
                names.append(name)
            except Exception as e:
                print(f"[ERROR] Failed to load known face {filename} for user {username}: {e}")
                
    return encodings, names


# ------------ LIVENESS --------------
_detector = None
_predictor = None


def get_detector():
    global _detector
    if _detector is None:
        print("[INFO] Initializing dlib face detector...")
        _detector = dlib.get_frontal_face_detector()
    return _detector


def get_predictor():
    global _predictor
    if _predictor is None:
        print(f"[INFO] Loading dlib shape predictor from {LANDMARK_MODEL}...")
        if not os.path.exists(LANDMARK_MODEL):
            raise FileNotFoundError(
                f"Landmark model file not found at '{LANDMARK_MODEL}'. "
                f"Please ensure this 99MB file is uploaded/present in the backend folder."
            )
        _predictor = dlib.shape_predictor(LANDMARK_MODEL)
    return _predictor


def compute_ear(eye) -> float:
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 2
EAR_OPEN_THRESH = 0.27
EAR_CLOSED_THRESH = 0.23

# ------------ ATTENDANCE ------------


def mark_attendance(username: str, name: str) -> None:
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    safe_user = _get_safe_username(username)
    excel_file = f"attendance_{safe_user}.xlsx"

    if not os.path.exists(excel_file):
        df = pd.DataFrame(columns=["name", "date", "time"])
        df.to_excel(excel_file, index=False)

    df = pd.read_excel(excel_file)

    if not ((df["name"] == name) & (df["date"] == current_date)).any():
        new_row = pd.DataFrame(
            {"name": [name], "date": [current_date], "time": [current_time]}
        )
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(excel_file, index=False)
        print(f"[{username}] {name} marked present at {current_time}")
    else:
        print(f"[{username}] Attendance already marked for {name} today.")


class AttendanceRow(BaseModel):
    name: str
    date: str
    time: str


# ------------ ROUTES ----------------


@app.get("/api/attendance", response_model=list[AttendanceRow])
def get_attendance(username: str = Depends(_verify_token)):
    safe_user = _get_safe_username(username)
    excel_file = f"attendance_{safe_user}.xlsx"
    if not os.path.exists(excel_file):
        return []
    df = pd.read_excel(excel_file)
    rows = [
        AttendanceRow(
            name=str(row["name"]),
            date=row["date"].strftime("%Y-%m-%d")
            if isinstance(row["date"], datetime)
            else str(row["date"]),
            time=str(row["time"]),
        )
        for _, row in df.iterrows()
    ]
    return rows


@app.get("/api/attendance/export")
def export_attendance(username: str = Depends(_verify_token)):
    safe_user = _get_safe_username(username)
    excel_file = f"attendance_{safe_user}.xlsx"
    if not os.path.exists(excel_file):
        df = pd.DataFrame(columns=["name", "date", "time"])
        df.to_excel(excel_file, index=False)

    return FileResponse(
        path=excel_file,
        media_type=(
            "application/"
            "vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        filename=f"attendance_{safe_user}.xlsx",
    )


@app.post("/api/faces/register")
async def register_face(
    name: str = Form(...),
    photo: UploadFile = File(...),
    username: str = Depends(_verify_token)
):
    ext = os.path.splitext(photo.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        return {"status": "error", "message": "Only JPG/PNG allowed"}

    safe_user = _get_safe_username(username)
    user_dir = os.path.join(IMAGE_DIR, safe_user)
    os.makedirs(user_dir, exist_ok=True)

    save_path = os.path.join(user_dir, f"{name}{ext}")
    with open(save_path, "wb") as f:
        f.write(await photo.read())

    return {"status": "ok", "message": f"Registered {name}"}


@app.post("/api/recognize")
async def recognize_sequence(
    photos: List[UploadFile] = File(...),
    username: str = Depends(_verify_token)
):
    if not photos:
        return {"recognized": [], "liveness": []}

    known_encodings, known_names = load_known_faces_for_user(username)

    all_face_encodings: list[list[np.ndarray]] = []
    frame_ears: list[float | None] = []

    for idx, photo in enumerate(photos):
        image_bytes = await photo.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if frame is None:
            print(f"[ERROR] Could not decode frame {idx}")
            all_face_encodings.append([])
            frame_ears.append(None)
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = get_detector()(gray_frame, 0)

        all_face_encodings.append(face_encodings)

        ear_value = None
        if len(faces) > 0:
            shape = get_predictor()(gray_frame, faces[0])
            left_eye_points = [
                (shape.part(i).x, shape.part(i).y) for i in range(36, 42)
            ]
            right_eye_points = [
                (shape.part(i).x, shape.part(i).y) for i in range(42, 48)
            ]
            left_ear = compute_ear(left_eye_points)
            right_ear = compute_ear(right_eye_points)
            ear_value = float((left_ear + right_ear) / 2.0)

        frame_ears.append(ear_value)

    print("[DEBUG] EARS per frame:", frame_ears)
    print("[DEBUG] known faces:", len(known_encodings))

    name = "Unknown"
    if known_encodings:
        best_overall_distance = 1.0
        best_overall_name = "Unknown"
        for frame_encodings in all_face_encodings:
            for face_encoding in frame_encodings:
                distances = face_recognition.face_distance(
                    known_encodings, face_encoding
                )
                best_idx = int(np.argmin(distances))
                best_distance = float(distances[best_idx])
                if best_distance < best_overall_distance:
                    best_overall_distance = best_distance
                    best_overall_name = known_names[best_idx]

        if best_overall_distance < 0.6:
            name = best_overall_name
            print(
                f"[DEBUG] Best match {name} with distance "
                f"{best_overall_distance:.3f}"
            )
        else:
            print(
                f"[DEBUG] No good match, best distance "
                f"{best_overall_distance:.3f}"
            )

    liveness_ok = False
    if name != "Unknown":
        states: list[str] = []
        for ear in frame_ears:
            if ear is None:
                states.append("none")
            elif ear >= EAR_OPEN_THRESH:
                states.append("open")
            elif ear <= EAR_CLOSED_THRESH:
                states.append("closed")
            else:
                states.append("mid")

        print("[DEBUG] EAR states:", states)

        has_open = any(s == "open" for s in states)
        has_closed = any(s == "closed" for s in states)

        if has_open and has_closed:
            liveness_ok = True

        print(
            f"[DEBUG] Liveness pattern -> "
            f"has_open={has_open}, has_closed={has_closed}, "
            f"liveness_ok={liveness_ok}"
        )

    if name != "Unknown" and liveness_ok:
        mark_attendance(username, name)
    elif name != "Unknown":
        print(f"[INFO] {name} recognized but liveness NOT confirmed")

    if name == "Unknown":
        return {"recognized": [], "liveness": []}
    return {"recognized": [name], "liveness": [liveness_ok]}


@app.get("/api/status")
def status(username: str = Depends(_verify_token)):
    safe_user = _get_safe_username(username)
    encodings, names = load_known_faces_for_user(username)
    excel_file = f"attendance_{safe_user}.xlsx"
    return {
        "known_faces": len(names),
        "attendance_file": os.path.exists(excel_file),
    }


# ------------ ADMIN: ATTENDANCE MANAGEMENT (Protected) ----------


@app.delete("/api/attendance/{index}")
def delete_attendance_row(index: int, username: str = Depends(_verify_token)):
    """Delete a single attendance row by its index."""
    safe_user = _get_safe_username(username)
    excel_file = f"attendance_{safe_user}.xlsx"
    if not os.path.exists(excel_file):
        raise HTTPException(status_code=404, detail="No attendance file")

    df = pd.read_excel(excel_file)
    if index < 0 or index >= len(df):
        raise HTTPException(status_code=400, detail="Index out of range")

    df = df.drop(index).reset_index(drop=True)
    df.to_excel(excel_file, index=False)
    return {"status": "ok", "remaining": len(df)}


@app.delete("/api/attendance")
def clear_attendance(username: str = Depends(_verify_token)):
    """Clear all attendance records."""
    safe_user = _get_safe_username(username)
    excel_file = f"attendance_{safe_user}.xlsx"
    df = pd.DataFrame(columns=["name", "date", "time"])
    df.to_excel(excel_file, index=False)
    return {"status": "ok", "message": "All attendance records cleared"}


@app.get("/api/faces/list")
def list_registered_faces(username: str = Depends(_verify_token)):
    """List all registered face image filenames for this user."""
    safe_user = _get_safe_username(username)
    user_dir = os.path.join(IMAGE_DIR, safe_user)
    os.makedirs(user_dir, exist_ok=True)
    faces = []
    for filename in os.listdir(user_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(filename)[0]
            faces.append({"filename": filename, "name": name})
    return faces


@app.delete("/api/faces/{face_name}")
def delete_registered_face(face_name: str, username: str = Depends(_verify_token)):
    """Delete a registered face by name."""
    safe_user = _get_safe_username(username)
    user_dir = os.path.join(IMAGE_DIR, safe_user)
    os.makedirs(user_dir, exist_ok=True)
    deleted = False
    for filename in os.listdir(user_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(filename)[0]
            if name.lower() == face_name.lower():
                os.remove(os.path.join(user_dir, filename))
                deleted = True
                break

    if not deleted:
        raise HTTPException(status_code=404, detail="Face not found")

    return {"status": "ok", "message": f"Deleted {face_name}"}


# ------------ ENTRY POINT -----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"[INFO] Starting server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)