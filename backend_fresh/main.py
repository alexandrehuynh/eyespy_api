from fastapi import FastAPI, File, UploadFile
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from collections import deque
from fastapi.responses import FileResponse

app = FastAPI()

# Directories for saving uploaded and processed videos (relative to the project root)
BASE_DIR = Path(__file__).resolve().parent.parent  # Navigate one level up
UPLOAD_FOLDER = BASE_DIR / "uploads"
PROCESSED_FOLDER = BASE_DIR / "processed"

# Ensure folders exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
PROCESSED_FOLDER.mkdir(exist_ok=True)

# Mediapipe Pose initialization
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Constants for smoothing and motion trails
SMOOTHING_WINDOW = 5
MOTION_TRAIL_LENGTH = 10

# Initialize smoothing history and motion trails
smoothing_history = {i: deque(maxlen=SMOOTHING_WINDOW) for i in range(33)}  # 33 landmarks
motion_trails = {i: deque(maxlen=MOTION_TRAIL_LENGTH) for i in range(33)}

def smooth_landmarks(landmarks, history, index):
    """Apply moving average smoothing for landmarks."""
    history[index].append(landmarks)
    if len(history[index]) < SMOOTHING_WINDOW:
        return landmarks  # Not enough data for smoothing
    return np.mean(history[index], axis=0)

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    input_path = UPLOAD_FOLDER / file.filename
    output_path = PROCESSED_FOLDER / f"processed_{file.filename}"

    with open(input_path, "wb") as f:
        f.write(file.file.read())

    cap = cv2.VideoCapture(str(input_path))
    out = None

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Smooth landmarks
                smoothed_landmarks = [
                    smooth_landmarks(
                        [lm.x, lm.y, lm.z, lm.visibility],
                        smoothing_history,
                        i
                    ) for i, lm in enumerate(landmarks)
                ]

                # Calculate pelvis as origin
                left_hip = np.array(smoothed_landmarks[23][:3])  # Left hip index
                right_hip = np.array(smoothed_landmarks[24][:3])  # Right hip index
                pelvis_origin = (left_hip + right_hip) / 2

                # Adjust landmarks relative to the pelvis origin
                adjusted_landmarks = [
                    [
                        lm[0] - pelvis_origin[0],
                        lm[1] - pelvis_origin[1],
                        lm[2] - pelvis_origin[2],
                        lm[3]
                    ]
                    if lm[3] > 0.5 else None  # Handle occlusion by skipping low-visibility landmarks
                    for lm in smoothed_landmarks
                ]

                # Visualize motion trails and landmarks
                for i, lm in enumerate(adjusted_landmarks):
                    if lm is not None:
                        x, y = int(lm[0] * frame.shape[1]), int(lm[1] * frame.shape[0])
                        motion_trails[i].append((x, y))
                        for j in range(1, len(motion_trails[i])):
                            cv2.line(frame, motion_trails[i][j - 1], motion_trails[i][j], (0, 255, 0), 2)

                # Draw 3D axes at the pelvis origin
                origin_x, origin_y = int(pelvis_origin[0] * frame.shape[1]), int(pelvis_origin[1] * frame.shape[0])
                cv2.line(frame, (origin_x, origin_y), (origin_x + 50, origin_y), (0, 0, 255), 3)  # X-axis (Red)
                cv2.line(frame, (origin_x, origin_y), (origin_x, origin_y - 50), (0, 255, 0), 3)  # Y-axis (Green)
                cv2.line(frame, (origin_x, origin_y), (origin_x, origin_y + 50), (255, 0, 0), 3)  # Z-axis (Blue)

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Write output video
            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))
            out.write(frame)

    cap.release()
    if out:
        out.release()

    return FileResponse(output_path, media_type="video/mp4", filename=output_path.name)