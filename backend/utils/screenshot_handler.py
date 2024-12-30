import os
import cv2

def get_fps(video_path):
    """Retrieve the FPS (frames per second) of the video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30  # Default to 30 FPS if FPS is invalid

def capture_screenshots(video_path, joint_data, output_folder, interval_seconds=5):
    """
    Captures screenshots based on selected frames and generates metadata.
    
    Args:
        video_path (str): Path to the input video file.
        joint_data (dict): Dictionary containing joint data for each frame.
        output_folder (str): Folder to save screenshots and metadata.
        interval_seconds (int): Interval in seconds between selected frames.
    
    Returns:
        dict: Metadata for screenshots including filenames, frame numbers, timestamps, and joint data.
    """
    # Create output subfolder for screenshots
    screenshots_folder = os.path.join(output_folder, "screenshots")
    os.makedirs(screenshots_folder, exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    fps = get_fps(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_metadata = []

    # Determine frame selection
    selected_frames = select_frames(total_frames, fps, interval_seconds=interval_seconds)

    frame_number = 0
    while frame_number < total_frames:
        if frame_number in selected_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_number / fps
            screenshot_name = f"frame_{frame_number:03d}.png"
            screenshot_path = os.path.join(screenshots_folder, screenshot_name)

            # Save the screenshot
            try:
                cv2.imwrite(screenshot_path, frame)
            except Exception as e:
                print(f"Error saving screenshot {screenshot_name}: {e}")
                continue

            # Add metadata for the screenshot, including all joint data
            metadata = {
                "file_name": screenshot_name,
                "frame_number": frame_number,
                "timestamp": f"{int(timestamp // 60):02}:{int(timestamp % 60):02}",
                "joint_data": joint_data.get(frame_number, {})
            }
            frame_metadata.append(metadata)

        frame_number += int(fps * interval_seconds)

    cap.release()
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