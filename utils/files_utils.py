import os
import uuid
from datetime import datetime

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