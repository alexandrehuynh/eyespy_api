from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .models import PoseEstimationResponse, ProcessingStatus, ConfidenceMetrics, Keypoint
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
    """
    video_path = None
    try:
        print(f"Received video: {video.filename}, size: {video.size} bytes")
        
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
        print("Saving uploaded file...")
        video_path = await video_processor.save_upload(video)
        if not video_path:
            raise HTTPException(
                status_code=500,
                detail="Failed to save video file"
            )
        
        print(f"Video saved to: {video_path}")
        
        # Extract frames
        print("Extracting frames...")
        frames, video_metadata = await video_processor.extract_frames(
            video_path,
            target_fps=target_fps,
            max_duration=max_duration
        )
        
        print(f"Frame extraction complete. Frames extracted: {len(frames)}")
        print(f"Video metadata: {video_metadata}")
        
        if not frames:
            raise HTTPException(
                status_code=400,
                detail=f"No frames could be extracted from video: {video_metadata.get('error', 'Unknown error')}"
            )
        
        # Process frames
        print("Processing frames for pose estimation...")
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
        avg_confidence = 0.0
        
        if latest_keypoints:
            for kp in latest_keypoints:
                confidence_by_part[kp.name] = kp.confidence
            avg_confidence = sum(kp.confidence for kp in latest_keypoints) / len(latest_keypoints)
        
        confidence_metrics = ConfidenceMetrics(
            average_confidence=avg_confidence,
            keypoints=confidence_by_part
        )
        
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
        print(f"Error processing video: {str(e)}")
        if video_path:
            try:
                await video_processor.cleanup(video_path)
            except Exception as cleanup_error:
                print(f"Error during cleanup: {str(cleanup_error)}")
                
        return PoseEstimationResponse(
            status=ProcessingStatus.FAILED,
            error=str(e)
        )