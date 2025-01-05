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
    """Process the video, overlay landmarks and angles, and save the result."""
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

            if results.pose_landmarks:
                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                )

                # Extract landmarks
                landmarks = results.pose_landmarks.landmark

                # Get coordinates for specific joints
                # Upper body (shoulder-elbow-wrist)
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Lower body (hip-knee-ankle)
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Spine alignment (shoulder-hip-knee)
                spine_top = shoulder
                spine_mid = hip
                spine_bottom = knee

                # Calculate angles
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                knee_angle = calculate_angle(hip, knee, ankle)
                hip_angle = calculate_angle(shoulder, hip, knee)
                spine_angle = calculate_angle(spine_top, spine_mid, spine_bottom)

                # Visualize angles
                cv2.putText(image, f'Elbow: {int(elbow_angle)}',
                            tuple(np.multiply(elbow, [image.shape[1], image.shape[0]]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f'Knee: {int(knee_angle)}',
                            tuple(np.multiply(knee, [image.shape[1], image.shape[0]]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f'Hip: {int(hip_angle)}',
                            tuple(np.multiply(hip, [image.shape[1], image.shape[0]]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f'Spine: {int(spine_angle)}',
                            tuple(np.multiply(hip, [image.shape[1] + 50, image.shape[0] - 50]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

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

