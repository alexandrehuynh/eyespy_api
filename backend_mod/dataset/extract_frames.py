import cv2
import os

def extract_frames(video_path, output_dir, label):
    """
    Extracts frames from a video and saves them with the given label.
    Args:
        video_path (str): Path to the uploaded video.
        output_dir (str): Directory to save extracted frames.
        label (str): Label to apply to the frames.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Save frame with label and frame count
        frame_path = os.path.join(output_dir, f"{label}_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Frames extracted and saved in {output_dir}")