import os
import cv2
import mediapipe as mp
import numpy as np
from utils.pose_landmarks import calculate_joint_angles, POSE_LANDMARKS
from PIL import Image, ImageDraw, ImageFont

def process_video(input_path, output_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose.Pose()

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Load a TrueType font for Pillow
    font_path = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"  # Use a compatible .ttf font
    font = ImageFont.truetype(font_path, 20)  # Font size can be adjusted dynamically

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame with Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                # Calculate joint angles
                angles = calculate_joint_angles(landmarks)

                # Convert frame to a Pillow image
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)

                # Overlay small angle symbols and numbers near joints
                for joint, angle in angles.items():
                    if joint in POSE_LANDMARKS:
                        joint_index = POSE_LANDMARKS[joint]
                        if joint_index >= len(landmarks) or landmarks[joint_index].visibility < 0.5:
                            continue  # Skip joints that are not visible

                        joint_coords = landmarks[joint_index]
                        joint_x = int(joint_coords.x * frame.shape[1])
                        joint_y = int(joint_coords.y * frame.shape[0])

                        # Draw angle symbol near the joint
                        draw.text((joint_x + 10, joint_y - 10), f"∠{int(angle)}°", font=font, fill=(255, 255, 0))

                # Display angle table in the top-right corner
                start_x, start_y = 10, 20  # Starting position for the table
                draw.rectangle([(start_x, start_y), (start_x + 200, start_y + 20 * (len(angles) + 2))], fill=(0, 0, 0))
                draw.text((start_x + 5, start_y + 5), "Joint    Angle", font=font, fill=(255, 255, 255))
                table_y = start_y + 30
                for joint, angle in angles.items():
                    draw.text((start_x + 5, table_y), f"{joint:<12} {int(angle)}°", font=font, fill=(255, 255, 255))
                    table_y += 20

                # Convert back to OpenCV format
                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            out.write(frame)
    finally:
        cap.release()
        out.release()

        # Clean up the uploaded file
        if os.path.exists(input_path):
            os.remove(input_path)