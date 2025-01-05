import os
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from backend_with_models.utils import (
    calculate_joint_angles,
    POSE_LANDMARKS,
    capture_screenshots,
    save_screenshot_metadata,
    predict_squat,
    calculate_deviation, 
    calculate_angle
)

# Initialize Mediapipe components
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose.Pose()

# Load font for annotations
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_SIZE = 20
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

def read_video(input_path):
    """Open video and return capture object."""
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, frame_width, frame_height, fps

def write_video(output_path, frame_width, frame_height, fps):
    """Initialize video writer."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def detect_pose(frame):
    """Detect pose landmarks using Mediapipe."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp_pose.process(rgb_frame)


def render_pose_landmarks(frame, landmarks):
    """Draw Mediapipe skeleton on the frame."""
    mp_drawing.draw_landmarks(
        frame,
        landmarks,
        mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )

def render_joint_angles(frame, landmarks, angles):
    """Overlay joint angles near their corresponding landmarks."""
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    frame_width, frame_height = frame.shape[1], frame.shape[0]
    for joint, angle in angles.items():
        joint_index = POSE_LANDMARKS[joint]
        joint_coords = landmarks[joint_index]
        joint_x = int(joint_coords.x * frame_width)
        joint_y = int(joint_coords.y * frame_height)
        draw.text((joint_x + 10, joint_y - 10), f"{int(angle)}°", font=font, fill=(255, 255, 0))

    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

def process_frame(frame):
    """Process a single frame: detect pose, calculate angles, and overlay annotations."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_pose.process(rgb_frame)

    if not results.pose_landmarks:
        return frame, None

    landmarks = results.pose_landmarks.landmark

    # Draw skeleton
    render_pose_landmarks(frame, results.pose_landmarks)

    # Calculate joint angles
    angles = calculate_joint_angles(landmarks)

    # Overlay joint angles
    frame = render_joint_angles(frame, landmarks, angles)

    # NEW: Calculate specific deviations and additional angles
    frame_width = frame.shape[1]
    knee_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * frame_width,
         landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * frame.shape[0]],
        [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * frame_width,
         landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * frame.shape[0]],
        [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * frame_width,
         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * frame.shape[0]]
    )

    shoulder_deviation = calculate_deviation(
        [frame_width // 2, 0],  # Midline of the frame
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame_width,
         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]]
    )

    # Overlay new annotations
    cv2.putText(frame, f"Knee Angle: {knee_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Shoulder Deviation: {shoulder_deviation:.2f}px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, angles
def process_frame_with_squat_detection(frame):
    """
    Process a single frame: detect pose, calculate angles, detect squats, and overlay annotations.
    Args:
        frame (numpy.ndarray): The input video frame.
    Returns:
        tuple: Processed frame with annotations, joint angles, and squat type.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_pose.process(rgb_frame)

    if not results.pose_landmarks:
        return frame, None, None

    landmarks = results.pose_landmarks.landmark

    # Draw skeleton
    render_pose_landmarks(frame, results.pose_landmarks)

    # Calculate joint angles
    angles = calculate_joint_angles(landmarks)

    # Overlay joint angles
    frame = render_joint_angles(frame, landmarks, angles)

    # Detect squat type
    landmarks_flat = [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
    squat_type_index = predict_squat(landmarks_flat)
    squat_labels = {0: "Good Squat", 1: "Bad Squat", 2: "No Squat"}
    squat_type = squat_labels.get(squat_type_index, "Unknown")

    # Annotate frame with squat type
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    draw.text((50, 50), f"Squat Type: {squat_type}", font=font, fill=(0, 255, 0))
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    return frame, angles, squat_type


def process_video_pipeline(input_path, output_path):
    """Orchestrates video processing: read, process frames, write output, and save screenshots."""
    cap, frame_width, frame_height, fps = read_video(input_path)
    out = write_video(output_path, frame_width, frame_height, fps)

    base_filename = os.path.splitext(os.path.basename(input_path))[0].replace("processed_", "")
    screenshot_folder = os.path.join("project_test_tools", "test_data", base_filename, "screenshots")
    os.makedirs(screenshot_folder, exist_ok=True)

    joint_data = {}
    squat_data = {}  # Store squat types for each frame
    processed_frames = {}  # Store annotated frames for screenshots
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame for pose detection, joint angles, and squat detection
            processed_frame, angles, squat_type = process_frame_with_squat_detection(frame)

            if angles:
                joint_data[frame_count] = angles

            if squat_type:
                squat_data[frame_count] = squat_type

            # Store processed frame
            processed_frames[frame_count] = processed_frame

            # Write processed frame to the video
            out.write(processed_frame)
            frame_count += 1

        # Capture screenshots and generate metadata
        screenshot_metadata = capture_screenshots(
            video_path=input_path,
            processed_frames=processed_frames,  # Pass frames with angles, skeletons, and squat types
            joint_data=joint_data,
            output_folder=os.path.join("project_test_tools", "test_data", base_filename),
            interval_seconds=4
        )

        # Save screenshot metadata
        save_screenshot_metadata(screenshot_metadata, os.path.join("project_test_tools", "test_data", base_filename))

        print(f"Processing complete. Processed video saved to: {output_path}")
        print(f"Screenshots and metadata saved to: {os.path.join('project_test_tools', 'test_data', base_filename)}")

    finally:
        mp_pose.close()  # Release Mediapipe resources
        cap.release()
        out.release()