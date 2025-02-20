from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .models import PoseEstimationResponse, ProcessingStatus, ConfidenceMetrics, Keypoint
from .video import VideoProcessor
from .pose.mediapipe_estimator import MediaPipeEstimator
from .pose.movenet_estimator import MovenetEstimator
from .pose.fusion import PoseFusion
import time
from typing import Optional, Dict, List
import psutil
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

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

@app.get("/")
async def root():
    return {"message": "Welcome to EyeSpy API"}

@app.post("/api/v1/pose", response_model=PoseEstimationResponse)
async def process_video(
    video: UploadFile = File(...),
    target_fps: Optional[int] = None,
    max_duration: Optional[int] = None,
    batch_size: int = 50  # Allow batch size to be configurable
) -> PoseEstimationResponse:
    """
    Process video in streaming mode with performance monitoring.
    """
    processing_start = time.time()
    video_processor = VideoProcessor(batch_size=batch_size)
    pose_estimator = MediaPipeEstimator()
    movenet_estimator = MovenetEstimator()
    pose_fusion = PoseFusion()
    
    # Performance metrics
    processed_frames = 0
    batch_times: List[float] = []
    memory_usage: List[float] = []
    
    try:
        video_path = await video_processor.save_upload(video)
        if not video_path:
            raise HTTPException(status_code=400, detail="Unable to save video")
        
        # Process frames in streaming mode
        async for frames_chunk, chunk_metadata in video_processor.extract_frames(
            video_path,
            target_fps=target_fps
        ):
            if not frames_chunk:
                continue
            
            batch_start = time.time()
            
            # Process chunk with both models in parallel
            try:
                # Run both models concurrently
                mediapipe_task = pose_estimator.process_frames(frames_chunk)
                movenet_task = movenet_estimator.process_frames(frames_chunk)
                
                mp_keypoints, mn_keypoints = await asyncio.gather(
                    mediapipe_task,
                    movenet_task
                )
                
                # Fuse results for each frame in the chunk
                fused_results = []
                for mp_kp, mn_kp in zip(mp_keypoints, mn_keypoints):
                    fused_kp = await pose_fusion.fuse_predictions(mp_kp, mn_kp)
                    fused_results.append(fused_kp)
                
                # Update metrics
                processed_frames += len(fused_results)
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # Monitor memory
                memory_percent = psutil.Process().memory_percent()
                memory_usage.append(memory_percent)
                
                # Log performance metrics
                fps = len(frames_chunk) / batch_time
                logger.info(
                    f"Batch processed: {processed_frames} frames total, "
                    f"Batch FPS: {fps:.1f}, "
                    f"Memory usage: {memory_percent:.1f}%"
                )
                
                # Free memory explicitly
                del frames_chunk
                del mp_keypoints
                del mn_keypoints
                
            except Exception as batch_error:
                logger.error(f"Error processing batch: {str(batch_error)}")
                continue
            
            # Optional: Add delay if memory usage is too high
            if memory_percent > 80:  # Arbitrary threshold
                logger.warning("High memory usage detected, adding small delay")
                await asyncio.sleep(0.1)
        
        # Calculate final performance metrics
        total_time = time.time() - processing_start
        avg_fps = processed_frames / total_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_memory = sum(memory_usage) / len(memory_usage)
        
        performance_metrics = {
            "total_frames": processed_frames,
            "total_time_seconds": total_time,
            "average_fps": avg_fps,
            "average_batch_time": avg_batch_time,
            "average_memory_percent": avg_memory,
            "peak_memory_percent": max(memory_usage),
            "batch_size": batch_size
        }
        
        logger.info(f"Processing complete. Metrics: {performance_metrics}")
        
        return PoseEstimationResponse(
            status=ProcessingStatus.COMPLETED,
            metadata={
                "performance": performance_metrics,
                "frames_processed": processed_frames,
                "processing_time": total_time
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return PoseEstimationResponse(
            status=ProcessingStatus.FAILED,
            error=str(e),
            metadata={"processing_time": time.time() - processing_start}
        )
    finally:
        # Cleanup
        if video_path:
            await video_processor.cleanup(video_path)