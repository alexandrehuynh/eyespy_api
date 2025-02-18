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
        
        # Extract and process frames
        print("Extracting and processing frames...")
        all_frames = []
        final_metadata = {}
        
        # Use async for to properly handle the generator
        async for frames_chunk, chunk_metadata in video_processor.extract_frames(
            video_path,
            target_fps=target_fps,
            max_duration=max_duration
        ):
            if not frames_chunk and chunk_metadata.get("error"):
                raise HTTPException(
                    status_code=400,
                    detail=chunk_metadata["error"]
                )
            
            all_frames.extend(frames_chunk)
            final_metadata.update(chunk_metadata)
        
        if not all_frames:
            raise HTTPException(
                status_code=400,
                detail="No frames could be extracted from video"
            )
        
        print(f"Frames extracted: {len(all_frames)}")
        print(f"Video metadata: {final_metadata}")
        
        # Process frames for pose estimation
        print("Processing frames for pose estimation...")
        try:
            all_keypoints = await pose_estimator.process_frames(all_frames)
            
            # Track processing progress
            frames_processed = len(all_keypoints) if all_keypoints else 0
            frames_total = len(all_frames)
            
            processing_metadata = {
                "frames_processed": frames_processed,
                "frames_total": frames_total,
                "processing_rate": f"{(frames_processed/frames_total)*100:.1f}%"
            }
            
            # Early exit if no keypoints detected
            if not all_keypoints:
                return PoseEstimationResponse(
                    status=ProcessingStatus.COMPLETED,
                    metadata={
                        **processing_metadata,
                        "error": "No keypoints were detected during processing"
                    }
                )
            
            # Warn if not all frames were processed
            if frames_processed < frames_total:
                print(f"Warning: Only processed {frames_processed}/{frames_total} frames")
            
            # Filter with additional validation
            valid_keypoints = [
                kp for kp in all_keypoints 
                if kp is not None and len(kp) > 0 and any(
                    keypoint.confidence > settings.GLOBAL_CONFIDENCE_THRESHOLD 
                    for keypoint in kp
                )
            ]
            
            # Handle case with no valid keypoints
            if not valid_keypoints:
                return PoseEstimationResponse(
                    status=ProcessingStatus.COMPLETED,
                    metadata={
                        **processing_metadata,
                        "error": "No valid keypoints met confidence threshold",
                        "keypoints_detected": len(all_keypoints),
                        "keypoints_valid": 0
                    }
                )
            
            latest_keypoints = valid_keypoints[-1]
            
            # Add success metadata
            processing_metadata.update({
                "keypoints_detected": len(all_keypoints),
                "keypoints_valid": len(valid_keypoints),
                "confidence_rate": f"{(len(valid_keypoints)/len(all_keypoints))*100:.1f}%"
            })
            
        except Exception as e:
            print(f"Error during frame processing: {str(e)}")
            return PoseEstimationResponse(
                status=ProcessingStatus.FAILED,
                error=f"Frame processing failed: {str(e)}",
                metadata={
                    "frames_processed": len(all_keypoints) if 'all_keypoints' in locals() else 0,
                    "frames_total": len(all_frames),
                    "error_type": type(e).__name__
                }
            )
        
        # Clean up the temporary file
        await video_processor.cleanup(video_path)
        
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
                **processing_metadata,
                "filename": video.filename,
                "frames_with_pose": len(valid_keypoints),
                "detection_rate": len(valid_keypoints) / len(all_frames)
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