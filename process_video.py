import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils import (
    calculate_joint_angles,
    POSE_LANDMARKS,
    get_subject_bbox,
    get_table_dimensions,
    adjust_table_position,
    draw_table_on_frame,
    clean_up_file,
)

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
    font_path = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
    font = ImageFont.truetype(font_path, 20)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_height, frame_width, _ = frame.shape

            # Process frame with Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get subject bounding box
                bbox = get_subject_bbox(landmarks, frame_width, frame_height)

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

                # Overlay angles near joints
                for joint, angle in angles.items():
                    if joint in POSE_LANDMARKS:
                        joint_index = POSE_LANDMARKS[joint]
                        joint_coords = landmarks[joint_index]
                        joint_x = int(joint_coords.x * frame_width)
                        joint_y = int(joint_coords.y * frame_height)
                        draw.text((joint_x + 10, joint_y - 10), f"∠{int(angle)}°", font=font, fill=(255, 255, 0))

                # Set table size and position
                table_width, table_height = get_table_dimensions(frame_width, frame_height)
                table_x, table_y = adjust_table_position(bbox, frame_width, frame_height, table_width, table_height)

                # Draw the table on the frame
                draw_table_on_frame(draw, font, angles, table_x, table_y)

                # Convert back to OpenCV format
                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            out.write(frame)
    finally:
        cap.release()
        out.release()

        # Clean up the uploaded file
        clean_up_file(input_path)