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

def get_table_dimensions(frame_width, frame_height, table_width_ratio=1/3, table_height_ratio=1/6):
    """Calculate table dimensions based on frame size."""
    table_width = int(frame_width * table_width_ratio)
    table_height = int(frame_height * table_height_ratio)
    return table_width, table_height

def draw_table_on_frame(draw, font, angles, start_x=10, start_y=10):
    """
    Draw a clean and readable table of joint angles on the frame using Pillow.

    Args:
        draw (ImageDraw.Draw): Pillow drawing object.
        font (ImageFont.FreeTypeFont): Font for text rendering.
        angles (dict): Dictionary of joint names and angles.
        start_x (int): X-coordinate for the top-left corner of the table.
        start_y (int): Y-coordinate for the top-left corner of the table.
    """
    # Define table header and column properties
    header = ["Joint", "Angle"]
    column_widths = [100, 60]  # Reduced width for "Angle" column
    row_height = 35           # Slightly larger rows for better spacing

    # Calculate table dimensions
    table_width = sum(column_widths)
    table_height = row_height * (len(angles) + 1)  # +1 for the header row

    # Draw the background rectangle for the table
    draw.rectangle([(start_x, start_y), (start_x + table_width, start_y + table_height)], fill=(0, 0, 0, 180))

    # Draw header row with gridlines
    header_y = start_y
    for i, col in enumerate(header):
        col_start_x = start_x + sum(column_widths[:i])
        text_x = col_start_x + 10  # Padding inside column
        draw.text((text_x, header_y + 8), col, font=font, fill=(255, 255, 255))
        # Draw vertical gridline for the column
        if i > 0:  # Skip the first column's left edge
            draw.line([(col_start_x, start_y), (col_start_x, start_y + table_height)], fill=(255, 255, 255), width=1)

    # Draw rows and horizontal gridlines
    row_y = start_y + row_height
    for joint, angle in angles.items():
        # Draw joint name
        text_joint_x = start_x + 10  # Padding inside "Joint" column
        draw.text((text_joint_x, row_y + 8), joint, font=font, fill=(255, 255, 255))

        # Draw angle
        text_angle_x = start_x + column_widths[0] + 10  # Start of "Angle" column
        draw.text((text_angle_x, row_y + 8), f"{int(angle)}°", font=font, fill=(255, 255, 255))

        # Draw a horizontal gridline for the row
        draw.line([(start_x, row_y), (start_x + table_width, row_y)], fill=(255, 255, 255), width=1)

        row_y += row_height

    # Draw the last vertical gridline (right edge of the table)
    draw.line([(start_x + table_width, start_y), (start_x + table_width, start_y + table_height)], fill=(255, 255, 255), width=1)

    # Draw the bottom horizontal gridline (bottom edge of the table)
    draw.line([(start_x, start_y + table_height), (start_x + table_width, start_y + table_height)], fill=(255, 255, 255), width=1)
    
def adjust_table_position(bbox, frame_width, frame_height, table_width, table_height, padding=10):
    """Adjust the table position to avoid overlapping the subject."""
    if bbox:
        subject_x1, subject_y1, subject_x2, subject_y2 = bbox

        # Default position (top-left corner)
        table_x = padding
        table_y = padding

        # If the table overlaps the subject, move it down
        if table_x < subject_x2 and table_y < subject_y2:
            table_y = subject_y2 + padding  # Move below the subject

        return table_x, table_y
    return padding, padding  # Default position