from fastapi import FastAPI, File, UploadFile
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
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

def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def process_video(input_path: Path, output_path: Path):
    """Process the video, overlay landmarks, angles, skeleton, dashed lines, and shaded areas."""
    cap = cv2.VideoCapture(str(input_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, int(cap.get(cv2.CAP_PROP_FPS)), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            height, width, _ = image.shape

            if results.pose_landmarks:
                # Draw Mediapipe skeleton
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Joint points
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),  # Connections
                )

                landmarks = results.pose_landmarks.landmark

                # Key points
                left_shoulder = np.array([
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height
                ])
                right_shoulder = np.array([
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height
                ])
                left_hip = np.array([
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height
                ])
                right_hip = np.array([
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * width,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * height
                ])
                left_elbow = np.array([
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height
                ])
                left_wrist = np.array([
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height
                ])
                left_knee = np.array([
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height
                ])
                left_ankle = np.array([
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width,
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height
                ])

                # Midpoints for spine alignment
                mid_shoulders = (left_shoulder + right_shoulder) / 2
                mid_hips = (left_hip + right_hip) / 2

                # Solid lines for pose structure
                cv2.line(image, tuple(left_shoulder.astype(int)), tuple(right_shoulder.astype(int)), (255, 0, 0), 2)  # Shoulders
                cv2.line(image, tuple(left_hip.astype(int)), tuple(right_hip.astype(int)), (0, 0, 255), 2)  # Hips
                cv2.line(image, tuple(mid_shoulders.astype(int)), tuple(mid_hips.astype(int)), (0, 255, 0), 2)  # Spine

                # Dashed vertical reference line
                for i in range(0, height, 20):
                    cv2.line(image, (int(width / 2), i), (int(width / 2), i + 10), (0, 255, 255), 2)

                # Dashed lines extending the spine
                for i in range(0, int(np.linalg.norm(mid_shoulders - mid_hips)), 20):
                    start = mid_shoulders + i * (mid_hips - mid_shoulders) / np.linalg.norm(mid_shoulders - mid_hips)
                    end = mid_shoulders + (i + 10) * (mid_hips - mid_shoulders) / np.linalg.norm(mid_shoulders - mid_hips)
                    cv2.line(image, tuple(start.astype(int)), tuple(end.astype(int)), (0, 255, 255), 2)

                # # Green shaded triangle for deviation visualization
                # deviation_vector = mid_shoulders - mid_hips
                # deviation_end = mid_shoulders + np.array([0, np.linalg.norm(deviation_vector)])
                # points = np.array([mid_hips, mid_shoulders, deviation_end], dtype=np.int32)
                # cv2.fillPoly(image, [points], color=(0, 255, 0, 50))

                # Joint angle calculations
                elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                spine_angle = calculate_angle(left_shoulder, left_hip, left_knee)

                # Display joint angles
                cv2.putText(image, f'Elbow: {int(elbow_angle)}°', tuple(left_elbow.astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(image, f'Knee: {int(knee_angle)}°', tuple(left_knee.astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(image, f'Hip: {int(hip_angle)}°', tuple(left_hip.astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(image, f'Spine: {int(spine_angle)}°', (int(left_hip[0]) + 50, int(left_hip[1]) - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Write the processed frame
            out.write(image)

    cap.release()
    out.release()

@app.post("/process-video/")
async def upload_video(file: UploadFile = File(...)):
    """Handle video upload and return the processed video."""
    input_path = UPLOAD_FOLDER / file.filename
    output_path = PROCESSED_FOLDER / f"processed_{file.filename}"

    # Save the uploaded file
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Process the video
    process_video(input_path, output_path)

    # Return the processed video file
    return FileResponse(output_path, media_type="video/mp4", filename=f"processed_{file.filename}")

