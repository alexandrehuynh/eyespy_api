from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .config import settings
from .models import PoseEstimationResponse, ProcessingStatus, ConfidenceMetrics, Keypoint, VideoRenderingResponse
from .video import VideoProcessor
from .video.renderer import VideoRenderer
from .pose.mediapipe_estimator import MediaPipeEstimator
import time
from typing import Optional, Dict, List, Any
import psutil
import logging
import asyncio
import numpy as np
import os
from pathlib import Path
from fastapi import Depends

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Create directories for rendered videos
output_dir = Path(settings.TEMP_DIR) / "rendered"
output_dir.mkdir(parents=True, exist_ok=True)

# Mount static files directory for serving rendered videos
app.mount("/videos", StaticFiles(directory=str(output_dir)), name="videos")

@app.get("/")
async def root():
    return {"message": "Welcome to EyeSpy API"}

@app.post("/api/v1/pose", response_model=PoseEstimationResponse)
async def process_video(
    video: UploadFile = File(...),
    target_fps: Optional[int] = None,
    max_duration: Optional[int] = None,
    performance_mode: str = "balanced"  # ["fast", "balanced", "quality"]
) -> PoseEstimationResponse:
    processing_start = time.time()
    processed_frames = 0
    batch_times = []
    memory_usage = []
    metadata = {}
    
    logger.info(f"Starting video processing with target_fps={target_fps}")
    
    try:
        video_processor = VideoProcessor(
            process_every_n_frames=2 if performance_mode == "fast" else 1,
            min_quality_threshold=0.4 if performance_mode == "fast" else 0.5
        )
        pose_estimator = MediaPipeEstimator()
        
        # Save uploaded video
        video_path = await video_processor.save_upload(video)
        if not video_path:
            logger.error("Failed to save uploaded video")
            raise HTTPException(status_code=400, detail="Unable to save video")
            
        logger.info(f"Video saved to: {video_path}")
        
        all_keypoints = []
        
        # Process frames in streaming mode
        async for frames_chunk, chunk_metadata in video_processor.extract_frames(
            video_path,
            target_fps=target_fps
        ):
            if not frames_chunk:
                logger.debug("Empty frames chunk received")
                continue
                
            metadata = chunk_metadata
            batch_start = time.time()
            
            try:
                logger.debug(f"Processing batch of {len(frames_chunk)} frames")
                # Process frames with MediaPipe
                keypoints = await pose_estimator.process_frames(frames_chunk)
                
                # Add valid keypoints to results
                for kp_list in keypoints:
                    if kp_list is not None:
                        all_keypoints.extend(kp_list)
                
                # Update metrics
                processed_frames += len(frames_chunk)
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                memory_usage.append(psutil.Process().memory_percent())
                
                # Log performance metrics
                fps = len(frames_chunk) / batch_time
                logger.info(
                    f"Batch processed: {processed_frames} frames total, "
                    f"Batch FPS: {fps:.1f}, "
                    f"Memory usage: {memory_usage[-1]:.1f}%"
                )
                
                # Free memory explicitly
                del frames_chunk
                del keypoints
                
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                continue
            
        # Calculate final metrics safely
        total_time = time.time() - processing_start
        avg_fps = processed_frames / total_time if total_time > 0 else 0
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0
        
        logger.info(f"Processing complete. Metrics: {avg_fps:.1f} FPS, {avg_batch_time:.3f} seconds per batch, {avg_memory:.1f}% memory usage")
        
        return PoseEstimationResponse(
            status=ProcessingStatus.COMPLETED if processed_frames > 0 else ProcessingStatus.FAILED,
            keypoints=all_keypoints if processed_frames > 0 else None,
            metadata={
                "processing_time": total_time,
                "frames_processed": processed_frames,
                "average_fps": avg_fps,
                "average_batch_time": avg_batch_time,
                "average_memory_usage": avg_memory,
                "video_info": {
                    "original_fps": metadata.get("original_fps", 0),
                    "total_frames": metadata.get("total_frames", 0),
                    "duration": metadata.get("duration", 0)
                }
            },
            confidence_metrics={
                "average_confidence": np.mean([kp.confidence for kp in all_keypoints if hasattr(kp, 'confidence')]) if all_keypoints else 0.0,
                "keypoints": {
                    "processed": processed_frames,
                    "detected": len(all_keypoints) if all_keypoints else 0
                }
            },
            validation_metrics=pose_estimator.frame_metadata,
            error="No frames processed successfully" if processed_frames == 0 else None
        )

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return PoseEstimationResponse(
            status=ProcessingStatus.FAILED,
            keypoints=None,
            metadata={"processing_time": time.time() - processing_start},
            confidence_metrics={"average_confidence": 0.0, "keypoints": {}},
            validation_metrics={},
            error=str(e)
        )
    finally:
        if video_processor:
            await video_processor.cleanup_resources()
        
        # Force garbage collection
        import gc
        gc.collect()

@app.post("/api/v1/render_video", response_model=VideoRenderingResponse)
async def render_video(
    video: UploadFile = File(...),
    background_tasks: BackgroundTasks = Depends(BackgroundTasks),
    target_fps: Optional[int] = Query(15, description="Target FPS for processing"),
    show_angles: bool = Query(True, description="Show joint angles in the output"),
    show_analytics: bool = Query(True, description="Show analytics text in the output"),
    performance_mode: str = Query("balanced", description="Performance mode (fast, balanced, quality)")
) -> VideoRenderingResponse:
    """
    Process a video and render the output with skeleton overlay and analytics
    """
    processing_start = time.time()
    
    try:
        # Initialize components
        video_processor = VideoProcessor(
            process_every_n_frames=2 if performance_mode == "fast" else 1,
            min_quality_threshold=0.4 if performance_mode == "fast" else 0.5
        )
        pose_estimator = MediaPipeEstimator()
        video_renderer = VideoRenderer()
        
        # Save uploaded video
        video_path = await video_processor.save_upload(video)
        if not video_path:
            logger.error("Failed to save uploaded video")
            raise HTTPException(status_code=400, detail="Unable to save video")
        
        logger.info(f"Video saved to: {video_path}")
        
        # Store frames and keypoints for rendering
        all_frames = []
        all_keypoints_per_frame = []
        metadata = {}
        
        # Process frames
        async for frames_chunk, chunk_metadata in video_processor.extract_frames(
            video_path,
            target_fps=target_fps
        ):
            if not frames_chunk:
                continue
                
            metadata = chunk_metadata
            
            # Process frames for pose estimation
            keypoints_batch = await pose_estimator.process_frames(frames_chunk)
            
            # Store frames and keypoints for rendering
            all_frames.extend(frames_chunk)
            all_keypoints_per_frame.extend(keypoints_batch)
        
        # If we have frames and keypoints, render the video
        if all_frames and all_keypoints_per_frame:
            # Generate output filename
            output_filename = f"rendered_{int(time.time())}.mp4"
            
            # Render video
            output_path = await video_renderer.render_video(
                all_frames,
                all_keypoints_per_frame,
                metadata,
                output_filename
            )
            
            # Calculate relative URL
            relative_url = f"/videos/{os.path.basename(output_path)}"
            
            # Clean up original video file and frames to save space
            background_tasks.add_task(video_processor.cleanup, video_path)
            
            return VideoRenderingResponse(
                status=ProcessingStatus.COMPLETED,
                video_url=relative_url,
                metadata={
                    "processing_time": time.time() - processing_start,
                    "frames_processed": len(all_frames),
                    "original_filename": video.filename,
                    "output_filename": output_filename,
                    "video_info": {
                        "fps": metadata.get("fps", 30),
                        "duration": metadata.get("duration", 0),
                        "width": all_frames[0].shape[1] if all_frames else 0,
                        "height": all_frames[0].shape[0] if all_frames else 0
                    }
                }
            )
        else:
            raise HTTPException(status_code=422, detail="No valid frames or keypoints detected")
    
    except Exception as e:
        logger.error(f"Error rendering video: {str(e)}")
        return VideoRenderingResponse(
            status=ProcessingStatus.FAILED,
            video_url=None,
            metadata={
                "processing_time": time.time() - processing_start
            },
            error=str(e)
        )
    finally:
        # Clean up resources
        if video_processor:
            await video_processor.cleanup_resources()
        
        # Force garbage collection
        import gc
        gc.collect()

@app.get("/api/v1/download_video/{filename}")
async def download_video(filename: str):
    """
    Download a rendered video file
    """
    file_path = output_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="video/mp4"
    )