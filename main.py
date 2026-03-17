import os
from datetime import datetime
from typing import List

import cv2
import dlib
import face_recognition
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from scipy.spatial import distance as dist

# ------------ CONFIG -----------------
IMAGE_DIR = "photos"
EXCEL_FILE = "attendance.xlsx"
LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat"

os.makedirs(IMAGE_DIR, exist_ok=True)

app = FastAPI()

# CORS so React (localhost:3000) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ FACE DATA -------------
known_encodings: List[np.ndarray] = []
known_names: List[str] = []


def load_known_faces() -> None:
    """Load encodings from IMAGE_DIR into memory."""
    global known_encodings, known_names
    known_encodings = []
    known_names = []

    for filename in os.listdir(IMAGE_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(IMAGE_DIR, filename)
            name = os.path.splitext(filename)[0]
            img = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(img)
            if not encodings:
                print(f"[WARN] No face found in {filename}")
                continue
            known_encodings.append(encodings[0])
            known_names.append(name)

    print(f"[INFO] Loaded {len(known_names)} known faces.")


load_known_faces()

# ------------ LIVENESS --------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(LANDMARK_MODEL)


def compute_ear(eye) -> float:
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Single-frame threshold kept for reference
EYE_AR_THRESH = 0.22      # below this = eyes closed
EYE_AR_CONSEC_FRAMES = 2  # unused here

# Multi-frame thresholds for blink-like pattern
EAR_OPEN_THRESH = 0.27    # >= this => eyes likely open
EAR_CLOSED_THRESH = 0.23  # <= this => eyes likely closed

# ------------ ATTENDANCE ------------


def mark_attendance(name: str) -> None:
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    if not os.path.exists(EXCEL_FILE):
        df = pd.DataFrame(columns=["name", "date", "time"])
        df.to_excel(EXCEL_FILE, index=False)

    df = pd.read_excel(EXCEL_FILE)

    if not ((df["name"] == name) & (df["date"] == current_date)).any():
        new_row = pd.DataFrame(
            {"name": [name], "date": [current_date], "time": [current_time]}
        )
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(EXCEL_FILE, index=False)
        print(f"{name} marked present at {current_time}")
    else:
        print(f"Attendance already marked for {name} today.")


class AttendanceRow(BaseModel):
    name: str
    date: str
    time: str


# ------------ ROUTES ----------------


@app.get("/api/attendance", response_model=list[AttendanceRow])
def get_attendance():
    """Return all attendance rows as JSON."""
    if not os.path.exists(EXCEL_FILE):
        return []
    df = pd.read_excel(EXCEL_FILE)
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
def export_attendance():
    """
    Download the attendance.xlsx file.
    Used by the frontend Export button.
    """
    if not os.path.exists(EXCEL_FILE):
        # create empty file if it doesn't exist yet
        df = pd.DataFrame(columns=["name", "date", "time"])
        df.to_excel(EXCEL_FILE, index=False)

    return FileResponse(
        path=EXCEL_FILE,
        media_type=(
            "application/"
            "vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        filename="attendance.xlsx",
    )


@app.post("/api/faces/register")
async def register_face(name: str = Form(...), photo: UploadFile = File(...)):
    """
    Save uploaded image into photos/ and refresh encodings.
    Frontend sends multipart/form-data with fields:
      - name: text
      - photo: file
    """
    # Save image
    ext = os.path.splitext(photo.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        return {"status": "error", "message": "Only JPG/PNG allowed"}

    save_path = os.path.join(IMAGE_DIR, f"{name}{ext}")
    with open(save_path, "wb") as f:
        f.write(await photo.read())

    # Re-load encodings
    load_known_faces()

    return {"status": "ok", "message": f"Registered {name}"}


@app.post("/api/recognize")
async def recognize_sequence(photos: List[UploadFile] = File(...)):
    """
    Multi-frame recognition with relaxed blink-like liveness.

    Frontend sends a short burst of frames as photos[].
    This endpoint:
      - Determines the best-matching known face across all frames.
      - Computes EAR per frame.
      - Classifies each frame as open / closed / mid / none.
      - Liveness is OK if we see at least one 'open' and at least one 'closed'
        frame in the sequence (order doesn't matter).
      - ONLY if liveness is OK AND a known face is found do we mark attendance.
    """
    if not photos:
        return {"recognized": [], "liveness": []}

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
        faces = detector(gray_frame, 0)

        all_face_encodings.append(face_encodings)

        # EAR for first detected face in this frame
        ear_value = None
        if len(faces) > 0:
            shape = predictor(gray_frame, faces[0])
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

    # 1) Identity: best match across all frames
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

        if best_overall_distance < 0.6:  # standard threshold
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

    # 2) Liveness: require both open and closed states somewhere in the sequence
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

    # 3) Mark attendance only if both identity and liveness are confirmed
    if name != "Unknown" and liveness_ok:
        mark_attendance(name)
    elif name != "Unknown":
        print(f"[INFO] {name} recognized but liveness NOT confirmed")

    # For compatibility with frontend, return arrays
    if name == "Unknown":
        return {"recognized": [], "liveness": []}
    return {"recognized": [name], "liveness": [liveness_ok]}


@app.get("/api/status")
def status():
    return {
        "known_faces": len(known_names),
        "attendance_file": os.path.exists(EXCEL_FILE),
    }