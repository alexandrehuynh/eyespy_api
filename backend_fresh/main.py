from fastapi import FastAPI, File, UploadFile
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from collections import deque, defaultdict
from fastapi.responses import FileResponse
from datetime import datetime
from mediapipe.python.solutions.pose import POSE_CONNECTIONS
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark, NormalizedLandmarkList
import json  # Import JSON for metadata saving
from filterpy.kalman import KalmanFilter # Add Kalman filtering for smoother tracking

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
    (start, end) for start, end in POSE_CONNECTIONS
    if 11 <= start < 33 and 11 <= end < 33  # Ensure indices are within valid range
])

# Constants for smoothing and motion trails
SMOOTHING_WINDOW = 5
MOTION_TRAIL_LENGTH = 10

# Initialize smoothing history and motion trails
smoothing_history = {i: deque(maxlen=SMOOTHING_WINDOW) for i in range(33)}  # 33 landmarks
motion_trails = {i: deque(maxlen=MOTION_TRAIL_LENGTH) for i in range(33)}

def smooth_landmarks(landmarks, history, index):
    """Apply moving average smoothing for landmarks."""
    if not all(isinstance(coord, (int, float)) for coord in landmarks[:3]):
        print(f"Skipping invalid landmark: {landmarks}")
        return landmarks  # Skip invalid landmarks

    history[index].append(landmarks[:3])  # Store only x, y, z for smoothing
    if len(history[index]) < SMOOTHING_WINDOW:
        return landmarks[:3]  # Return as is if not enough data
    return np.mean(history[index], axis=0)

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points in 3D space.
    a, b, c are points represented as [x, y, z].
    Returns the angle in degrees.
    """
    if a is None or b is None or c is None:
        return None  # Skip if any point is missing or occluded

    a = np.array(a[:3]).flatten()  # Use only x, y, z and ensure 1D array
    b = np.array(b[:3]).flatten()  # Ensure 1D array
    c = np.array(c[:3]).flatten()  # Ensure 1D array

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

def get_spine_midpoints(shoulder_midpoint, hip_midpoint, num_points=5):
    """
    Generate evenly spaced points along the line connecting the shoulder and hip midpoints.
    """
    spine_midpoints = []
    for i in range(num_points + 1):  # Include both endpoints
        t = i / num_points
        mid_x = (1 - t) * shoulder_midpoint[0] + t * hip_midpoint[0]
        mid_y = (1 - t) * shoulder_midpoint[1] + t * hip_midpoint[1]
        spine_midpoints.append((mid_x, mid_y))
    return spine_midpoints
    
def draw_arrow(frame, start_point, end_point, color=(0, 255, 255), thickness=2):
    """
    Draw an arrow between two points on the frame.
    
    Parameters:
    - frame: The image on which to draw.
    - start_point: The starting point of the arrow (x, y).
    - end_point: The ending point of the arrow (x, y).
    - color: The color of the arrow (default is yellow).
    - thickness: The thickness of the arrow line (default is 2).
    """
    cv2.arrowedLine(frame, start_point, end_point, color, thickness, tipLength=0.3)

def init_kalman_filters():
    """Initialize Kalman filters for each landmark"""
    filters = {}
    for i in range(33):
        kf = KalmanFilter(dim_x=6, dim_z=3)  # State: position, velocity, acceleration
        kf.F = np.array([[1, 1, 0.5, 0, 0, 0],    # State transition matrix
                        [0, 1, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0.5],
                        [0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 1]])
        filters[i] = kf
    return filters

def apply_kalman_filtering(landmark, kf):
    """Apply Kalman filtering to landmark coordinates."""
    if not isinstance(landmark, NormalizedLandmark):
        print(f"Invalid landmark detected: {landmark}")
        return None  # Skip invalid landmark

    measurement = np.array([landmark.x, landmark.y, landmark.z])
    kf.predict()
    kf.update(measurement)
    return kf.x[:3]  # Return filtered position

def draw_motion_heat_map(frame, motion_history):
    """Create a heat map showing movement intensity"""
    heat_map = np.zeros_like(frame)
    for positions in motion_history.values():
        for pos in positions:
            cv2.circle(heat_map, (int(pos[0]), int(pos[1])),
                      10, (0, 0, 255), -1)

    # Apply Gaussian blur to smooth the heat map
    heat_map = cv2.GaussianBlur(heat_map, (15, 15), 0)
    return cv2.addWeighted(frame, 0.7, heat_map, 0.3, 0)

def draw_velocity_vectors(frame, current_landmarks, previous_landmarks):
    """Draw velocity vectors for key joints"""
    if previous_landmarks:
        for i in range(len(current_landmarks)):
            if i in [11, 12, 13, 14, 23, 24, 25, 26]:  # Key joints
                curr = current_landmarks[i]
                prev = previous_landmarks[i]
                cv2.arrowedLine(frame,
                              (int(prev[0]), int(prev[1])),
                              (int(curr[0]), int(curr[1])),
                              (255, 0, 0), 2)

def create_3d_skeleton(landmarks):
    """Create a 3D skeleton visualization"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot joints
    xs = [lm[0] for lm in landmarks]
    ys = [lm[1] for lm in landmarks]
    zs = [lm[2] for lm in landmarks]
    ax.scatter(xs, ys, zs)

    # Draw connections
    for connection in CUSTOM_POSE_CONNECTIONS:
        start = landmarks[connection[0]]
        end = landmarks[connection[1]]
        ax.plot([start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]])

    return fig

def draw_analytics_overlay(frame, angles, velocities):
    """Draw an analytics overlay with joint angles and velocities."""
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 200), (0, 0, 0), -1)
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Add text for analytics
    y_offset = 40
    for joint, angle in angles.items():
        if angle is None:
            angle_text = "N/A"  # Default text for NoneType
        else:
            angle_text = f"{angle:.1f}°"
        
        cv2.putText(frame, f"{joint}: {angle_text}",
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        y_offset += 30
        
def draw_movement_paths(frame, motion_trails):
    """Draw paths of movement for key joints"""
    for joint_id, trail in motion_trails.items():
        points = np.array(list(trail))
        if len(points) >= 2:
            # Create smooth curve through points
            pts = points.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(frame, [pts], False, (0, 255, 255), 2)

            # Add direction indicators
            for i in range(len(points) - 1):
                if i % 2 == 0:  # Add arrows every other segment
                    pt1 = tuple(points[i].astype(int))
                    pt2 = tuple(points[i + 1].astype(int))
                    draw_arrow(frame, pt1, pt2, color=(0, 255, 255), thickness=2)

    
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
                        f"Elbow: {int(left_elbow_angle)}",
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
                        f"Knee: {int(left_knee_angle)}",
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
                        f"Hip: {int(left_hip_angle)}",
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

                # Calculate shoulder and hip midpoints
                left_shoulder = np.array([
                    smoothed_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0],
                    smoothed_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1],
                ])
                right_shoulder = np.array([
                    smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0],
                    smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1],
                ])
                left_hip = np.array([
                    smoothed_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0],
                    smoothed_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1],
                ])
                right_hip = np.array([
                    smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][0],
                    smoothed_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][1],
                ])

                # Midpoints for shoulders and hips
                shoulder_midpoint = (left_shoulder + right_shoulder) / 2
                hip_midpoint = (left_hip + right_hip) / 2

                # Get spine midpoints
                spine_midpoints = get_spine_midpoints(shoulder_midpoint, hip_midpoint, num_points=5)

                # Draw spine
                for i in range(len(spine_midpoints) - 1):
                    start_point = tuple(map(int, spine_midpoints[i]))
                    end_point = tuple(map(int, spine_midpoints[i + 1]))
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)  # Green line for spine

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
