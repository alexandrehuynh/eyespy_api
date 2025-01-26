from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .models import PoseEstimationResponse, ProcessingStatus
from .video import VideoProcessor
import time

app = FastAPI(
    title=settings.API_NAME,
    version=settings.VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to EyeSpy API"}

@app.post("/api/v1/pose", response_model=PoseEstimationResponse)
async def process_video(
    video: UploadFile = File(...)
) -> PoseEstimationResponse:
    # Validate video format
    if not any(video.filename.lower().endswith(fmt) 
              for fmt in settings.SUPPORTED_FORMATS):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Supported formats: {settings.SUPPORTED_FORMATS}"
        )
    
    video_processor = VideoProcessor()
    
    try:
        # Save uploaded file
        video_path = await video_processor.save_upload(video)
        if not video_path:
            raise HTTPException(
                status_code=500,
                detail="Failed to save video file"
            )
            
        # For now, just return a success response
        # We'll add actual processing in the next step
        return PoseEstimationResponse(
            status=ProcessingStatus.COMPLETED,
            metadata={
                "filename": video.filename,
                "saved_path": str(video_path)
            }
        )
        
    except Exception as e:
        return PoseEstimationResponse(
            status=ProcessingStatus.FAILED,
            error=str(e)
        )