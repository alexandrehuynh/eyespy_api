import os
import cv2
import mediapipe as mp
from calculations import calculate_angle

def process_video(input_path, output_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose.Pose()

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(rgb_frame)

        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            # Calculate joint angles
            landmarks = results.pose_landmarks.landmark
            left_knee_angle = calculate_angle(
                (landmarks[23].x, landmarks[23].y, landmarks[23].z),  # Hip
                (landmarks[25].x, landmarks[25].y, landmarks[25].z),  # Knee
                (landmarks[27].x, landmarks[27].y, landmarks[27].z)   # Ankle
            )
            right_knee_angle = calculate_angle(
                (landmarks[24].x, landmarks[24].y, landmarks[24].z),  # Hip
                (landmarks[26].x, landmarks[26].y, landmarks[26].z),  # Knee
                (landmarks[28].x, landmarks[28].y, landmarks[28].z)   # Ankle
            )

            # Overlay angles on video
            cv2.putText(frame, f"Left Knee: {int(left_knee_angle)}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f"Right Knee: {int(right_knee_angle)}", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    # Clean up the uploaded file
    if os.path.exists(input_path):
        os.remove(input_path)