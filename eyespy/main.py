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
import numpy as np

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
    batch_size: int = 50
) -> PoseEstimationResponse:
    processing_start = time.time()
    processed_frames = 0
    batch_times = []
    memory_usage = []
    metadata = {}
    fused_results = []  # Initialize fused_results list
    
    try:
        video_processor = VideoProcessor(batch_size=batch_size)
        pose_estimator = MediaPipeEstimator()
        movenet_estimator = MovenetEstimator()
        pose_fusion = PoseFusion()
        
        # Save uploaded video
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
                
            metadata = chunk_metadata
            batch_start = time.time()
            
            try:
                # Run both models concurrently
                mediapipe_task = pose_estimator.process_frames(frames_chunk)
                movenet_task = movenet_estimator.process_frames(frames_chunk)
                
                mp_keypoints, mn_keypoints = await asyncio.gather(
                    mediapipe_task,
                    movenet_task
                )
                
                # Fuse results for each frame in the chunk
                batch_results = []
                for mp_kp, mn_kp in zip(mp_keypoints, mn_keypoints):
                    fused_kp = await pose_fusion.fuse_predictions(mp_kp, mn_kp)
                    batch_results.append(fused_kp)
                
                fused_results.extend(batch_results)  # Add batch results to overall results
                
                # Update metrics
                processed_frames += len(batch_results)
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
                del mp_keypoints
                del mn_keypoints
                
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                continue
            
            # Optional: Add delay if memory usage is too high
            if memory_usage[-1] > 80:  # Arbitrary threshold
                logger.warning("High memory usage detected, adding small delay")
                await asyncio.sleep(0.1)
        
        # Calculate final metrics safely
        total_time = time.time() - processing_start
        avg_fps = processed_frames / total_time if total_time > 0 else 0
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0
        
        logger.info(f"Processing complete. Metrics: {avg_fps:.1f} FPS, {avg_batch_time:.3f} seconds per batch, {avg_memory:.1f}% memory usage")
        
        return PoseEstimationResponse(
            status="success" if processed_frames > 0 else "failed",
            keypoints=fused_results if processed_frames > 0 else None,
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
                "average_confidence": np.mean([kp.confidence for kp in fused_results]) if fused_results else 0.0,
                "keypoints": {
                    "processed": processed_frames,
                    "detected": len([k for k in fused_results if k is not None]) if fused_results else 0
                }
            },
            validation_metrics=pose_estimator.frame_metadata,
            error="No frames processed successfully" if processed_frames == 0 else None
        )

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return PoseEstimationResponse(
            status="failed",
            keypoints=None,
            metadata={"processing_time": time.time() - processing_start},
            confidence_metrics={"average_confidence": 0.0, "keypoints": {}},
            validation_metrics={},
            error=str(e)
        )
    finally:
        # Cleanup
        if video_path:
            await video_processor.cleanup(video_path)