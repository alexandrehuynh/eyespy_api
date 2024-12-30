import os
import json
from datetime import datetime
import uuid

def clean_up_file(file_path):
    """Delete a file if it exists."""
    if os.path.exists(file_path):
        os.remove(file_path)

def generate_unique_filename(original_filename, output_folder=None, use_uuid=False, timestamp_format="%m%d%Y_%H%M%S"):
    """Generate a unique output filename based on the original filename."""
    name, ext = os.path.splitext(original_filename)
    timestamp = datetime.now().strftime(timestamp_format)
    unique_id = f"_{uuid.uuid4().hex[:8]}" if use_uuid else ""
    filename = f"{name}_{timestamp}{unique_id}{ext}"
    
    if output_folder:
        return os.path.join(output_folder, filename)
    return filename

def save_file(file_content, destination_path):
    """Save binary file content to a specified destination."""
    with open(destination_path, "wb") as f:
        f.write(file_content)

def save_screenshot_metadata(metadata, output_folder):
    """
    Saves screenshot metadata to a JSON file in the specified output folder.

    Args:
        metadata (dict): Metadata dictionary to save.
        output_folder (str): Path to the folder where the metadata file will be saved.

    Returns:
        str: Path to the saved metadata file.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define the metadata file path
    metadata_file_path = os.path.join(output_folder, "screenshot_metadata.json")

    # Save the metadata to JSON
    with open(metadata_file_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    return metadata_file_path