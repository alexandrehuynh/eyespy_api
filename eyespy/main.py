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
    target_fps: int = 30,
    max_duration: int = 10
) -> PoseEstimationResponse:
    """
    Process video for pose estimation with multi-frame support
    - target_fps: Target frames per second to process
    - max_duration: Maximum duration in seconds to process
    """
    video_path = None
    try:
        # Validate video format
        if not any(video.filename.lower().endswith(fmt) 
                  for fmt in settings.SUPPORTED_FORMATS):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported video format. Supported formats: {settings.SUPPORTED_FORMATS}"
            )
        
        # Initialize processors
        video_processor = VideoProcessor()
        pose_estimator = MediaPipeEstimator()
        
        # Save uploaded file
        video_path = await video_processor.save_upload(video)
        if not video_path:
            raise HTTPException(
                status_code=500,
                detail="Failed to save video file"
            )
        
        # Extract frames
        frames, video_metadata = await video_processor.extract_frames(
            video_path,
            target_fps=target_fps,
            max_duration=max_duration
        )
        
        if not frames:
            raise HTTPException(
                status_code=400,
                detail="No frames could be extracted from video"
            )
        
        # Process all frames
        all_keypoints = await pose_estimator.process_frames(frames)
        
        # Filter out None values and get the most recent valid keypoints
        valid_keypoints = [kp for kp in all_keypoints if kp is not None]
        latest_keypoints = valid_keypoints[-1] if valid_keypoints else None
        
        # Clean up the temporary file
        await video_processor.cleanup(video_path)
        
        # If no valid keypoints were found
        if not valid_keypoints:
            return PoseEstimationResponse(
                status=ProcessingStatus.COMPLETED,
                metadata={
                    **video_metadata,
                    "filename": video.filename,
                    "message": "No pose detected in any frame"
                }
            )
        
        # Calculate confidence metrics
        confidence_by_part = {}
        for kp in latest_keypoints:
            confidence_by_part[kp.name] = kp.confidence
        
        # Calculate average confidence
        avg_confidence = sum(kp.confidence for kp in latest_keypoints) / len(latest_keypoints)
        
        confidence_metrics = {
            "average_confidence": avg_confidence,
            "keypoints": confidence_by_part
        }
        
        # Return successful response with keypoints and metrics
        return PoseEstimationResponse(
            status=ProcessingStatus.COMPLETED,
            keypoints=latest_keypoints,
            metadata={
                **video_metadata,
                "filename": video.filename,
                "frames_with_pose": len(valid_keypoints),
                "detection_rate": len(valid_keypoints) / len(frames)
            },
            confidence_metrics=confidence_metrics
        )
        
    except Exception as e:
        # Ensure cleanup happens even if there's an error
        if video_path:
            try:
                await video_processor.cleanup(video_path)
            except:
                pass  # Ignore cleanup errors in error handling
                
        # Return error response
        return PoseEstimationResponse(
            status=ProcessingStatus.FAILED,
            error=str(e)
        )