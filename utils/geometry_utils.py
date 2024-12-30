def get_subject_bbox(landmarks, frame_width, frame_height):
    """
    Calculate bounding box for the detected subject.

    Args:
        landmarks (list): List of Mediapipe landmarks.
        frame_width (int): Width of the video frame.
        frame_height (int): Height of the video frame.

    Returns:
        tuple: Bounding box as (min_x, min_y, max_x, max_y), or None if no visible landmarks.
    """
    x_coords = [lm.x for lm in landmarks if lm.visibility > 0.5]
    y_coords = [lm.y for lm in landmarks if lm.visibility > 0.5]

    if not x_coords or not y_coords:  # If no visible landmarks
        return None

    min_x = int(min(x_coords) * frame_width)
    max_x = int(max(x_coords) * frame_width)
    min_y = int(min(y_coords) * frame_height)
    max_y = int(max(y_coords) * frame_height)

    return (min_x, min_y, max_x, max_y)  # (x1, y1, x2, y2)