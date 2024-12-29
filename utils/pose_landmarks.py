from calculations import calculate_angle

POSE_LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

def calculate_joint_angles(landmarks):
    """
    Calculate angles for multiple joints using Mediapipe landmarks.
    Returns a dictionary of joint angles.
    """
    angles = {}

    # Elbows
    angles["left_elbow"] = calculate_angle(
        (landmarks[POSE_LANDMARKS["left_shoulder"]].x, landmarks[POSE_LANDMARKS["left_shoulder"]].y, landmarks[POSE_LANDMARKS["left_shoulder"]].z),
        (landmarks[POSE_LANDMARKS["left_elbow"]].x, landmarks[POSE_LANDMARKS["left_elbow"]].y, landmarks[POSE_LANDMARKS["left_elbow"]].z),
        (landmarks[POSE_LANDMARKS["left_wrist"]].x, landmarks[POSE_LANDMARKS["left_wrist"]].y, landmarks[POSE_LANDMARKS["left_wrist"]].z),
    )
    angles["right_elbow"] = calculate_angle(
        (landmarks[POSE_LANDMARKS["right_shoulder"]].x, landmarks[POSE_LANDMARKS["right_shoulder"]].y, landmarks[POSE_LANDMARKS["right_shoulder"]].z),
        (landmarks[POSE_LANDMARKS["right_elbow"]].x, landmarks[POSE_LANDMARKS["right_elbow"]].y, landmarks[POSE_LANDMARKS["right_elbow"]].z),
        (landmarks[POSE_LANDMARKS["right_wrist"]].x, landmarks[POSE_LANDMARKS["right_wrist"]].y, landmarks[POSE_LANDMARKS["right_wrist"]].z),
    )

    # Shoulders
    angles["left_shoulder"] = calculate_angle(
        (landmarks[POSE_LANDMARKS["left_hip"]].x, landmarks[POSE_LANDMARKS["left_hip"]].y, landmarks[POSE_LANDMARKS["left_hip"]].z),
        (landmarks[POSE_LANDMARKS["left_shoulder"]].x, landmarks[POSE_LANDMARKS["left_shoulder"]].y, landmarks[POSE_LANDMARKS["left_shoulder"]].z),
        (landmarks[POSE_LANDMARKS["left_elbow"]].x, landmarks[POSE_LANDMARKS["left_elbow"]].y, landmarks[POSE_LANDMARKS["left_elbow"]].z),
    )
    angles["right_shoulder"] = calculate_angle(
        (landmarks[POSE_LANDMARKS["right_hip"]].x, landmarks[POSE_LANDMARKS["right_hip"]].y, landmarks[POSE_LANDMARKS["right_hip"]].z),
        (landmarks[POSE_LANDMARKS["right_shoulder"]].x, landmarks[POSE_LANDMARKS["right_shoulder"]].y, landmarks[POSE_LANDMARKS["right_shoulder"]].z),
        (landmarks[POSE_LANDMARKS["right_elbow"]].x, landmarks[POSE_LANDMARKS["right_elbow"]].y, landmarks[POSE_LANDMARKS["right_elbow"]].z),
    )

    # Hips
    angles["left_hip"] = calculate_angle(
        (landmarks[POSE_LANDMARKS["left_knee"]].x, landmarks[POSE_LANDMARKS["left_knee"]].y, landmarks[POSE_LANDMARKS["left_knee"]].z),
        (landmarks[POSE_LANDMARKS["left_hip"]].x, landmarks[POSE_LANDMARKS["left_hip"]].y, landmarks[POSE_LANDMARKS["left_hip"]].z),
        (landmarks[POSE_LANDMARKS["left_shoulder"]].x, landmarks[POSE_LANDMARKS["left_shoulder"]].y, landmarks[POSE_LANDMARKS["left_shoulder"]].z),
    )
    angles["right_hip"] = calculate_angle(
        (landmarks[POSE_LANDMARKS["right_knee"]].x, landmarks[POSE_LANDMARKS["right_knee"]].y, landmarks[POSE_LANDMARKS["right_knee"]].z),
        (landmarks[POSE_LANDMARKS["right_hip"]].x, landmarks[POSE_LANDMARKS["right_hip"]].y, landmarks[POSE_LANDMARKS["right_hip"]].z),
        (landmarks[POSE_LANDMARKS["right_shoulder"]].x, landmarks[POSE_LANDMARKS["right_shoulder"]].y, landmarks[POSE_LANDMARKS["right_shoulder"]].z),
    )

    # Knees
    angles["left_knee"] = calculate_angle(
        (landmarks[POSE_LANDMARKS["left_hip"]].x, landmarks[POSE_LANDMARKS["left_hip"]].y, landmarks[POSE_LANDMARKS["left_hip"]].z),
        (landmarks[POSE_LANDMARKS["left_knee"]].x, landmarks[POSE_LANDMARKS["left_knee"]].y, landmarks[POSE_LANDMARKS["left_knee"]].z),
        (landmarks[POSE_LANDMARKS["left_ankle"]].x, landmarks[POSE_LANDMARKS["left_ankle"]].y, landmarks[POSE_LANDMARKS["left_ankle"]].z),
    )
    angles["right_knee"] = calculate_angle(
        (landmarks[POSE_LANDMARKS["right_hip"]].x, landmarks[POSE_LANDMARKS["right_hip"]].y, landmarks[POSE_LANDMARKS["right_hip"]].z),
        (landmarks[POSE_LANDMARKS["right_knee"]].x, landmarks[POSE_LANDMARKS["right_knee"]].y, landmarks[POSE_LANDMARKS["right_knee"]].z),
        (landmarks[POSE_LANDMARKS["right_ankle"]].x, landmarks[POSE_LANDMARKS["right_ankle"]].y, landmarks[POSE_LANDMARKS["right_ankle"]].z),
    )

    # Ankles
    angles["left_ankle"] = calculate_angle(
        (landmarks[POSE_LANDMARKS["left_knee"]].x, landmarks[POSE_LANDMARKS["left_knee"]].y, landmarks[POSE_LANDMARKS["left_knee"]].z),
        (landmarks[POSE_LANDMARKS["left_ankle"]].x, landmarks[POSE_LANDMARKS["left_ankle"]].y, landmarks[POSE_LANDMARKS["left_ankle"]].z),
        (landmarks[POSE_LANDMARKS["left_ankle"]].x + 1, landmarks[POSE_LANDMARKS["left_ankle"]].y, landmarks[POSE_LANDMARKS["left_ankle"]].z),  # Artificial point for alignment
    )
    angles["right_ankle"] = calculate_angle(
        (landmarks[POSE_LANDMARKS["right_knee"]].x, landmarks[POSE_LANDMARKS["right_knee"]].y, landmarks[POSE_LANDMARKS["right_knee"]].z),
        (landmarks[POSE_LANDMARKS["right_ankle"]].x, landmarks[POSE_LANDMARKS["right_ankle"]].y, landmarks[POSE_LANDMARKS["right_ankle"]].z),
        (landmarks[POSE_LANDMARKS["right_ankle"]].x + 1, landmarks[POSE_LANDMARKS["right_ankle"]].y, landmarks[POSE_LANDMARKS["right_ankle"]].z),  # Artificial point for alignment
    )

    return angles