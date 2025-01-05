import os
import cv2

def get_fps(video_path):
    """Retrieve the FPS (frames per second) of the video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30  # Default to 30 FPS if FPS is invalid

def capture_screenshots(video_path, processed_frames, joint_data, output_folder, interval_seconds=4):
    """
    Captures screenshots from processed frames with Mediapipe landmarks and annotations.

    Args:
        video_path (str): Path to the input video file.
        processed_frames (dict): Dictionary of annotated frames keyed by frame number.
        joint_data (dict): Joint data for each frame.
        output_folder (str): Folder to save screenshots and metadata.
        interval_seconds (int): Interval in seconds between selected frames.

    Returns:
        dict: Metadata for screenshots including filenames, frame numbers, timestamps, and joint data.
    """
    screenshots_folder = os.path.join(output_folder, "screenshots")
    os.makedirs(screenshots_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    frame_interval = int(fps * interval_seconds)
    frame_metadata = []

    for frame_number, processed_frame in processed_frames.items():
        if frame_number % frame_interval == 0:
            timestamp = frame_number / fps
            screenshot_name = f"frame_{frame_number:03d}.png"
            screenshot_path = os.path.join(screenshots_folder, screenshot_name)
            cv2.imwrite(screenshot_path, processed_frame)

            metadata = {
                "file_name": screenshot_name,
                "frame_number": frame_number,
                "timestamp": f"{int(timestamp // 60):02}:{int(timestamp % 60):02}",
                "joint_data": joint_data.get(frame_number, {})
            }
            frame_metadata.append(metadata)

    return {"screenshots": frame_metadata}

def select_frames(total_frames, fps, interval_seconds=5):
    """
    Select frames dynamically based on video FPS.

    Args:
        total_frames (int): Total number of frames in the video.
        fps (float): Frames per second of the video.
        interval_seconds (int): Interval in seconds between selected frames.

    Returns:
        set: Selected frame numbers.
    """
    frame_interval = int(fps * interval_seconds)  # Calculate frame interval dynamically
    return set(range(0, total_frames, frame_interval))