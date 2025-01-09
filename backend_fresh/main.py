from fastapi import FastAPI, File, UploadFile
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from collections import deque
from fastapi.responses import FileResponse
from datetime import datetime
from mediapipe.python.solutions.pose import POSE_CONNECTIONS
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
import json  # Import JSON for metadata saving

app = FastAPI()

# Directories for saving uploaded and processed videos (relative to the project root)
BASE_DIR = Path(__file__).resolve().parent.parent  # Navigate one level up
UPLOAD_FOLDER = BASE_DIR / "uploads"
PROCESSED_FOLDER = BASE_DIR / "processed"
META_FOLDER = PROCESSED_FOLDER / "meta"

# Ensure folders exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
PROCESSED_FOLDER.mkdir(exist_ok=True)

# Mediapipe Pose initialization
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Define custom pose connections (excluding face)
CUSTOM_POSE_CONNECTIONS = set([
    connection for connection in POSE_CONNECTIONS
    if connection[0] >= 11 and connection[1] >= 11  # Keep only body-related connections (landmarks 11+)
])

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

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points in 3D space.
    a, b, c are points represented as [x, y, z].
    Returns the angle in degrees.
    """
    if a is None or b is None or c is None:
        return None  # Skip if any point is missing or occluded

    a = np.array(a[:3])  # Use only x, y, z (:3) (ignore visibility)
    b = np.array(b[:3])  # For only x, y use (:2)
    c = np.array(c[:3])

    # Vectors BA and BC
    ba = a - b
    bc = c - b

    # Dot product and magnitudes
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    # Prevent division by zero
    if magnitude_ba == 0 or magnitude_bc == 0:
        return None

    # Calculate the angle in radians and convert to degrees
    radians = np.arccos(dot_product / (magnitude_ba * magnitude_bc))
    angle = np.degrees(radians)

    return angle

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    input_path = UPLOAD_FOLDER / file.filename

    # Generate a timestamp for the processed file
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    output_filename = f"processed_{timestamp}_{file.filename}"
    output_metadata_path = META_FOLDER / f"{timestamp}_metadata.json"
    output_path = PROCESSED_FOLDER / output_filename

    with open(input_path, "wb") as f:
        f.write(file.file.read())

    cap = cv2.VideoCapture(str(input_path))
    out = None
    metadata = {"frames": []}  # Metadata for storing angles and timestamps

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                height, width, _ = frame.shape

                # Smooth landmarks
                smoothed_landmarks = [
                    smooth_landmarks(
                        [lm.x * width, lm.y * height, lm.z * width, lm.visibility],
                        smoothing_history,
                        i,
                    ) for i, lm in enumerate(landmarks)
                ]

                # Define joint groups for angles
                left_elbow_angle = calculate_angle(
                    smoothed_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                    smoothed_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                    smoothed_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                )
                left_knee_angle = calculate_angle(
                    smoothed_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                    smoothed_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                    smoothed_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                )
                left_hip_angle = calculate_angle(
                    smoothed_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                    smoothed_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                    smoothed_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                )

                # Save angles to metadata
                metadata["frames"].append({
                    "timestamp": cap.get(cv2.CAP_PROP_POS_MSEC),
                    "angles": {
                        "elbow": left_elbow_angle,
                        "knee": left_knee_angle,
                        "hip": left_hip_angle,
                    },
                })

                # Display angles on video with dynamic positioning
                if left_elbow_angle is not None:
                    cv2.putText(
                        frame,
                        f"Elbow: {int(left_elbow_angle)}°",
                        (int(smoothed_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][0] + 20),
                         int(smoothed_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][1] - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0) if left_elbow_angle < 90 else (0, 0, 255),
                        2,
                    )
                if left_knee_angle is not None:
                    cv2.putText(
                        frame,
                        f"Knee: {int(left_knee_angle)}°",
                        (int(smoothed_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value][0] + 20),
                         int(smoothed_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value][1] - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0) if left_knee_angle < 90 else (0, 0, 255),
                        2,
                    )
                if left_hip_angle is not None:
                    cv2.putText(
                        frame,
                        f"Hip: {int(left_hip_angle)}°",
                        (int(smoothed_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0] + 20),
                         int(smoothed_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1] - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0) if left_hip_angle < 90 else (0, 0, 255),
                        2,
                    )

                # Calculate pelvis origin
                left_hip = np.array(smoothed_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][:2])
                right_hip = np.array(smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][:2])
                pelvis_origin = (left_hip + right_hip) / 2

                # Draw extended and semi-transparent 3D axes at the pelvis origin
                origin_x, origin_y = int(pelvis_origin[0]), int(pelvis_origin[1])
                overlay = frame.copy()

                # Extended axes
                x_length, y_length, z_length = int(width * 0.2), int(height * 0.2), int(height * 0.2)
                cv2.line(overlay, (origin_x, origin_y), (origin_x + x_length, origin_y), (0, 0, 255), 3)
                cv2.line(overlay, (origin_x, origin_y), (origin_x, origin_y - y_length), (0, 255, 0), 3)
                cv2.line(overlay, (origin_x, origin_y), (origin_x, origin_y + z_length), (255, 0, 0), 3)

                # Apply opacity to axes
                alpha = 0.5
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Create a filtered landmark list without facial landmarks
                filtered_landmarks = NormalizedLandmarkList()
                filtered_landmarks.landmark.extend(
                    [lm for i, lm in enumerate(results.pose_landmarks.landmark) if i >= 11]
                )

                # Remap connections for the filtered landmarks
                adjusted_connections = [
                    (start - 11, end - 11) for start, end in POSE_CONNECTIONS
                    if start >= 11 and end >= 11  # Only keep connections for landmarks 11+
                ]

                # Draw landmarks and connections excluding facial landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    filtered_landmarks,  # Use the filtered landmark set
                    adjusted_connections,  # Use the remapped connections
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                )

            # Write output video
            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, cap.get(cv2.CAP_PROP_FPS),
                                      (frame.shape[1], frame.shape[0]))
            out.write(frame)

    cap.release()
    if out:
        out.release()


    # Ensure the metadata directory exists
    metadata_dir = output_metadata_path.parent
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata to a JSON file
    with open(output_metadata_path, "w") as meta_file:
        json.dump(metadata, meta_file)

    return FileResponse(output_path, media_type="video/mp4", filename=output_filename)