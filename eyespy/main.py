from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .models import PoseEstimationResponse, ProcessingStatus
from .video import VideoProcessor
from .pose.mediapipe_estimator import MediaPipeEstimator
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
    video: UploadFile = File(...),
    max_frames: int = 30
) -> PoseEstimationResponse:
    # Validate video format
    if not any(video.filename.lower().endswith(fmt) 
              for fmt in settings.SUPPORTED_FORMATS):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Supported formats: {settings.SUPPORTED_FORMATS}"
        )
    
    video_processor = VideoProcessor()
    pose_estimator = MediaPipeEstimator()
    
    try:
        # Save uploaded file
        video_path = await video_processor.save_upload(video)
        if not video_path:
            raise HTTPException(
                status_code=500,
                detail="Failed to save video file"
            )
        
        # Extract frames
        frames, fps = await video_processor.extract_frames(video_path, max_frames)
        
        if not frames:
            raise HTTPException(
                status_code=400,
                detail="No frames could be extracted from video"
            )
        
        # Process middle frame for now (we'll add multi-frame processing later)
        middle_frame = frames[len(frames)//2]
        keypoints = pose_estimator.process_frame(middle_frame)
        
        # Cleanup
        await video_processor.cleanup(video_path)
        
        if not keypoints:
            return PoseEstimationResponse(
                status=ProcessingStatus.COMPLETED,
                metadata={
                    "filename": video.filename,
                    "frames_extracted": len(frames),
                    "fps": fps,
                    "message": "No pose detected in frame"
                }
            )
        
        return PoseEstimationResponse(
            status=ProcessingStatus.COMPLETED,
            keypoints=keypoints,
            metadata={
                "filename": video.filename,
                "frames_extracted": len(frames),
                "fps": fps
            }
        )
        
    except Exception as e:
        if video_path:
            await video_processor.cleanup(video_path)
        return PoseEstimationResponse(
            status=ProcessingStatus.FAILED,
            error=str(e)
        )