from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
from process_video import process_video_pipeline
from utils import ( generate_unique_filename, save_file, clean_up_file )

app = FastAPI()

UPLOAD_DIR = "uploads/"
PROCESSED_DIR = "processed/"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

@app.post("/process-video/")
async def process_video_endpoint(file: UploadFile = File(...)):
    try:
        # Generate unique filenames for input and output
        unique_name = generate_unique_filename(file.filename)
        input_path = os.path.join(UPLOAD_DIR, unique_name)
        output_path = os.path.join(PROCESSED_DIR, f"processed_{unique_name}")

        # Save the uploaded file
        save_file(await file.read(), input_path)

        # Process the video
        process_video_pipeline(input_path, output_path)

        # Return the processed video as a downloadable file
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"processed_{file.filename}"
        )
    finally:
        # Clean up uploaded file after processing
        clean_up_file(input_path)