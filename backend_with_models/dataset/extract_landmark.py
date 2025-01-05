import mediapipe as mp
import cv2
import os
import csv

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def extract_landmarks(frame_dir, output_csv):
    """
    Extracts pose landmarks from frames and saves them in a CSV file.
    Args:
        frame_dir (str): Directory containing frames.
        output_csv (str): Path to save the output CSV file.
    """
    # Open CSV file for writing
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Header: Landmark_0_x, Landmark_0_y, ..., Landmark_32_z
        writer.writerow(["Label"] + [f"Landmark_{i}_{axis}" for i in range(33) for axis in ["x", "y", "z"]])

        # Process frames
        for frame_file in os.listdir(frame_dir):
            frame_path = os.path.join(frame_dir, frame_file)
            label = frame_file.split("_")[0]  # Extract label from filename

            # Process with MediaPipe
            frame = cv2.imread(frame_path)
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                writer.writerow([label] + landmarks)

    print(f"Landmarks saved to {output_csv}")