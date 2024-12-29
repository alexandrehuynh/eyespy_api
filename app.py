from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
from process_video import process_video  
from datetime import datetime

app = FastAPI()

UPLOAD_DIR = "uploads/"
PROCESSED_DIR = "processed/"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def generate_unique_filename(original_name):
    """Generate a unique filename using the current timestamp."""
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")  # MMDDYYYY_HHMMSS
    name, ext = original_name.rsplit(".", 1)
    return f"{name}_{timestamp}.{ext}"

@app.post("/process-video/")
async def process_video_endpoint(file: UploadFile = File(...)):
    # Generate a unique filename for the uploaded video
    unique_name = generate_unique_filename(file.filename)
    input_path = os.path.join(UPLOAD_DIR, unique_name)
    output_path = os.path.join(PROCESSED_DIR, f"processed_{unique_name}")

    # Save the uploaded file
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Process the video
    process_video(input_path, output_path)

    # Return the processed video as a downloadable file
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"processed_{file.filename}"  # Suggests the filename for the client
    )