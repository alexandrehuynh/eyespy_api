def get_subject_bbox(landmarks, frame_width, frame_height):
    """Calculate bounding box for the detected subject."""
    x_coords = [lm.x for lm in landmarks if lm.visibility > 0.5]
    y_coords = [lm.y for lm in landmarks if lm.visibility > 0.5]

    if not x_coords or not y_coords:  # If no visible landmarks
        return None

    min_x = int(min(x_coords) * frame_width)
    max_x = int(max(x_coords) * frame_width)
    min_y = int(min(y_coords) * frame_height)
    max_y = int(max(y_coords) * frame_height)

    return (min_x, min_y, max_x, max_y)  # (x1, y1, x2, y2)

def get_table_dimensions(frame_width, frame_height, table_width_ratio=1/3, table_height_ratio=1/6):
    """Calculate table dimensions based on frame size."""
    table_width = int(frame_width * table_width_ratio)
    table_height = int(frame_height * table_height_ratio)
    return table_width, table_height

def adjust_table_position(bbox, frame_width, frame_height, table_width, table_height):
    """Adjust the table position to avoid overlapping the subject."""
    if bbox:
        subject_x1, subject_y1, subject_x2, subject_y2 = bbox

        # Default position (top-right corner)
        table_x = frame_width - table_width - 10
        table_y = 10

        # If the table overlaps the subject, move it down
        if table_x < subject_x2 and table_y < subject_y2:
            table_y = subject_y2 + 10  # Move below the subject

        return table_x, table_y
    return frame_width - table_width - 10, 10  # Default position