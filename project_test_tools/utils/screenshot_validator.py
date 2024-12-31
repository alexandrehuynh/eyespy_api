import os
import cv2
import json

def validate_screenshots(screenshots_folder, metadata_file, fps=30):
    """
    Validates screenshots against metadata.

    Args:
        screenshots_folder (str): Path to the folder containing screenshots.
        metadata_file (str): Path to the JSON metadata file.
        fps (int): Frames per second of the video (default is 30).

    Returns:
        dict: Summary of the validation process.
    """
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    screenshots_metadata = metadata.get("screenshots", [])
    metadata_by_frame = {entry["frame_number"]: entry for entry in screenshots_metadata}

    validation_results = {"aligned": [], "mismatched": [], "missing_metadata": []}

    # Iterate through screenshots
    for screenshot_file in os.listdir(screenshots_folder):
        if screenshot_file.endswith(".png"):
            # Extract frame number from screenshot file name
            frame_number = int(screenshot_file.split("_")[1].split(".")[0])

            # Check if metadata exists for the frame
            metadata_entry = metadata_by_frame.get(frame_number)
            if not metadata_entry:
                validation_results["missing_metadata"].append(screenshot_file)
                continue

            # Validate timestamp
            expected_timestamp = frame_number / fps
            actual_timestamp = metadata_entry.get("timestamp")
            expected_timestamp_formatted = f"{int(expected_timestamp // 60):02}:{int(expected_timestamp % 60):02}"
            if actual_timestamp != expected_timestamp_formatted:
                validation_results["mismatched"].append({
                    "screenshot": screenshot_file,
                    "frame_number": frame_number,
                    "expected_timestamp": expected_timestamp_formatted,
                    "actual_timestamp": actual_timestamp,
                })
                continue

            # Add to aligned results
            validation_results["aligned"].append(screenshot_file)

    return validation_results

