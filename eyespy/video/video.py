import logging
from fastapi import UploadFile
import time
import cv2
import numpy as np
import asyncio
from pathlib import Path
from typing import AsyncGenerator, List, Tuple, Dict, Optional
from collections import deque
from ..config import settings
from .quality import AdaptiveFrameQualityAssessor, QualityMetrics
from .frame_selector import FrameSelector
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameBuffer:
    """Thread-safe frame buffer for parallel processing"""
    def __init__(self, maxsize: int = 100):
        self.frames = deque(maxlen=maxsize)
        self.lock = asyncio.Lock()
        self.not_empty = asyncio.Event()
        self.completed = asyncio.Event()

    async def put(self, frame_data: dict):
        async with self.lock:
            self.frames.append(frame_data)
            self.not_empty.set()

    async def get(self) -> Optional[dict]:
        while True:
            async with self.lock:
                if self.frames:
                    return self.frames.popleft()
                if self.completed.is_set():
                    return None
                # Release lock while waiting
                self.not_empty.clear()
                
            # Wait outside the lock
            await self.not_empty.wait()

    def mark_completed(self):
        self.completed.set()

class VideoProcessor:
    def __init__(
        self,
        batch_size: int = 50,
        buffer_size: int = 100,
        min_quality_threshold: float = 0.5
    ):
        """Initialize VideoProcessor with configurable parameters"""
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_quality_threshold = min_quality_threshold
        self.frame_selector = FrameSelector()
        self.temp_dir = Path(settings.TEMP_DIR)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.quality_assessor = AdaptiveFrameQualityAssessor()
        self._cleanup_handlers_registered = False
        self.active_tasks = set()  # Track active tasks

    async def cleanup_resources(self):
        """Cleanup all resources"""
        logger.info("Cleaning up resources...")
        
        # Cancel all active tasks
        for task in self.active_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Clear the set
        self.active_tasks.clear()
        
        # Cleanup temporary files
        if self.temp_dir.exists():
            for temp_file in self.temp_dir.glob("temp_*"):
                try:
                    temp_file.unlink()
                    logger.debug(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    logger.error(f"Error removing temporary file {temp_file}: {e}")

        # Release other resources
        if hasattr(self, 'quality_assessor'):
            self.quality_assessor.__del__()
        
        logger.info("Cleanup completed")

    def __del__(self):
        """Ensure cleanup on deletion"""
        if asyncio.get_event_loop().is_running():
            asyncio.create_task(self.cleanup_resources())

    async def extract_frames(
        self,
        video_path: Path,
        target_fps: Optional[int] = None,
        max_duration: Optional[int] = None
    ) -> AsyncGenerator[Tuple[List[np.ndarray], dict], None]:
        """Extract frames with performance optimization"""
        if not video_path.exists():
            yield [], self._create_error_metadata("Video file not found")
            return

        cap = cv2.VideoCapture(str(video_path))
        try:
            metadata = await self._get_video_properties(cap)
            frame_buffer = asyncio.Queue(maxsize=self.buffer_size)
            result_buffer = asyncio.Queue(maxsize=self.buffer_size)
            
            # Use ThreadPoolExecutor for CPU-bound operations
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Start frame reader and quality assessor tasks
                reader_task = asyncio.create_task(
                    self._frame_reader(cap, frame_buffer, executor, metadata)
                )
                quality_task = asyncio.create_task(
                    self._quality_assessor(frame_buffer, result_buffer)
                )
                
                try:
                    while True:
                        result_data = await result_buffer.get()
                        if result_data is None:
                            break
                        
                        # Process frames efficiently
                        frames = result_data['frames']
                        if not frames:
                            continue
                            
                        # Use numpy operations for better performance
                        frames_array = np.array(frames)
                        yield frames_array.tolist(), {
                            **metadata,
                            "frames_processed": len(frames),
                            "quality_metrics": self._calculate_quality_stats(result_data['metrics'])
                        }
                        
                        # Allow other tasks to run
                        await asyncio.sleep(0)
                        
                finally:
                    # Cleanup tasks
                    for task in [reader_task, quality_task]:
                        if not task.done():
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
        finally:
            cap.release()

    async def _frame_reader(
        self,
        cap: cv2.VideoCapture,
        frame_buffer: asyncio.Queue,
        executor: ThreadPoolExecutor,
        metadata: Dict
    ) -> None:
        """Read frames from video and put them in buffer"""
        try:
            frame_count = 0
            logger.info("Starting frame reader...")
            
            # Get video properties
            fps = metadata.get('fps', 30.0)
            target_fps = metadata.get('target_fps', 30.0)
            frame_interval = metadata.get('frame_interval', 1)
            
            logger.info(f"Video properties: FPS={fps}, Target FPS={target_fps}, Frame Interval={frame_interval}")
            
            while cap.isOpened():
                ret, frame = await asyncio.get_event_loop().run_in_executor(
                    executor, cap.read
                )
                
                if not ret:
                    logger.info(f"Frame reading complete. Total frames read: {frame_count}")
                    break
                
                # Process frames at target FPS
                if frame_count % frame_interval == 0:
                    if frame is not None and isinstance(frame, np.ndarray):
                        # Log frame properties
                        logger.debug(f"Reading frame {frame_count}, shape: {frame.shape}, type: {frame.dtype}")
                        
                        # Ensure frame is in correct format
                        if frame.dtype != np.uint8:
                            frame = frame.astype(np.uint8)
                        
                        # Handle different color spaces
                        if len(frame.shape) == 2:  # Grayscale
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        elif frame.shape[2] == 4:  # RGBA
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                        
                        await frame_buffer.put({
                            'frame': frame.copy(),  # Create a copy to prevent reference issues
                            'index': frame_count,
                            'timestamp': frame_count / fps if fps > 0 else 0
                        })
                    else:
                        logger.warning(f"Invalid frame at index {frame_count}")
                
                frame_count += 1
                
                # Periodically yield control to other tasks
                if frame_count % 10 == 0:
                    await asyncio.sleep(0)
            
            logger.info(f"Frame reader finished. Total frames processed: {frame_count}")
            
        except Exception as e:
            logger.error(f"Error in frame reader: {str(e)}")
        finally:
            # Signal end of processing
            await frame_buffer.put(None)

    async def _process_batch(self, frames: List[np.ndarray]) -> List[QualityMetrics]:
        """Process a batch of frames with proper async handling"""
        return await asyncio.gather(
            *[self.quality_assessor.assess_frame(frame) for frame in frames]
        )

    async def _frame_selector(self, result_buffer: asyncio.Queue) -> Tuple[List[np.ndarray], Dict]:
        """Select frames based on quality and distribution with proper async handling"""
        frames = []
        indices = []
        metrics = []
        
        try:
            while True:
                result_data = await result_buffer.get()
                if result_data is None:
                    break
                
                frames.extend(result_data['frames'])
                indices.extend(result_data['indices'])
                metrics.extend(result_data['metrics'])
            
            if not frames:
                print("No frames collected for selection")
                return [], {}

            # Properly await the frame selection
            selected_frames, selected_indices, selection_stats = await self.frame_selector.select_best_frames(
                frames,
                metrics
            )
            
            stats = {
                'quality_stats': self._calculate_quality_stats(metrics),
                'selection_stats': selection_stats,
                'frame_indices': selected_indices
            }
            
            return selected_frames, stats
            
        except Exception as e:
            print(f"Frame selection error: {str(e)}")
            return [], {}

    async def _quality_assessor(self, frame_buffer: asyncio.Queue, result_buffer: asyncio.Queue):
        """Process frames for quality assessment"""
        try:
            frames_processed = 0
            frames_passed = 0
            logger.info("Starting quality assessor...")
            
            while True:
                batch_data = await frame_buffer.get()
                if batch_data is None:
                    logger.info(f"Quality assessment complete. Processed: {frames_processed}, Passed: {frames_passed}")
                    break
                
                if not isinstance(batch_data, dict) or 'frame' not in batch_data:
                    logger.error(f"Invalid batch data format: {batch_data}")
                    continue
                
                frames_processed += 1
                # Process single frame
                frame_metrics = await self.quality_assessor.assess_frame(batch_data['frame'])
                
                # Detailed metrics logging
                logger.info(f"""
Frame {batch_data['index']} Quality Metrics:
- Valid: {frame_metrics.is_valid}
- Overall Score: {frame_metrics.overall_score:.3f} (threshold: {self.min_quality_threshold})
- Brightness: {frame_metrics.brightness:.3f}
- Contrast: {frame_metrics.contrast:.3f}
- Blur Score: {frame_metrics.blur_score:.3f}
- Coverage Score: {frame_metrics.coverage_score:.3f}
                """)
                
                if frame_metrics.is_valid and frame_metrics.overall_score >= self.min_quality_threshold:
                    frames_passed += 1
                    logger.info(f"✅ Frame {batch_data['index']} PASSED quality check")
                    await result_buffer.put({
                        'frames': [batch_data['frame']],
                        'metrics': [frame_metrics],
                        'indices': [batch_data['index']]
                    })
                else:
                    failure_reason = "invalid frame" if not frame_metrics.is_valid else f"low score ({frame_metrics.overall_score:.3f} < {self.min_quality_threshold})"
                    logger.warning(f"❌ Frame {batch_data['index']} FAILED quality check: {failure_reason}")
                
        except Exception as e:
            logger.error(f"Error in quality assessor: {str(e)}")
        finally:
            await result_buffer.put(None)

    async def _perform_calibration(self, cap: cv2.VideoCapture) -> Optional[Dict]:
        """Perform initial calibration and return calibration data"""
        try:
            calibration_frames = []
            sample_interval = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.quality_assessor.calibration_size))
            frame_count = 0
            
            while len(calibration_frames) < self.quality_assessor.calibration_size:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Sample frames at regular intervals
                if frame_count % sample_interval == 0:
                    if frame is not None:
                        # Convert frame if necessary
                        if len(frame.shape) == 2:  # If grayscale
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        elif frame.shape[2] == 4:  # If RGBA
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                            
                        calibration_frames.append(frame)
                
                frame_count += 1
                if frame_count % 5 == 0:
                    await asyncio.sleep(0)
            
            print(f"Collected {len(calibration_frames)} frames for calibration")
            
            if calibration_frames:
                if self.quality_assessor.calibrate_thresholds(calibration_frames):
                    return {
                        'success': True,
                        'frames_sampled': len(calibration_frames),
                        'frame_count': frame_count
                    }
            
            print("Failed to collect enough calibration frames")
            return None
            
        except Exception as e:
            print(f"Calibration error: {str(e)}")
            return None

    async def _get_video_properties(self, cap: cv2.VideoCapture) -> Dict:
        """Get video properties with frame interval calculation"""
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Calculate frame interval based on target FPS
            original_fps = fps if fps > 0 else 30.0
            target_fps = self.target_fps if hasattr(self, 'target_fps') else 30.0
            frame_interval = max(1, int(original_fps / target_fps))
            
            metadata = {
                "fps": fps,
                "total_frames": total_frames,
                "duration": duration,
                "frame_interval": frame_interval,
                "original_fps": fps,
                "target_fps": target_fps
            }
            
            logger.info(f"Video properties: fps={fps}, frames={total_frames}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting video properties: {str(e)}")
            return {
                "fps": 0,
                "total_frames": 0,
                "duration": 0,
                "frame_interval": 1,
                "original_fps": 0,
                "target_fps": 30.0
            }

    async def _process_frames(
        self,
        cap: cv2.VideoCapture,
        target_frames: int,
        sample_interval: int
    ) -> Tuple[Dict, Dict]:
        """Process frames in batches with quality assessment"""
        frames = []
        quality_metrics = []
        frame_indices = []
        frame_count = 0
        
        while cap.isOpened() and len(frames) < target_frames:
            batch_frames = []
            batch_indices = []
            
            # Read batch of frames
            for _ in range(self.batch_size):
                if frame_count >= self.batch_size:
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_interval == 0:
                    batch_frames.append(frame)
                    batch_indices.append(frame_count)
                
                frame_count += 1
            
            if not batch_frames:
                break
            
            # Process batch
            batch_metrics = await self._process_batch(batch_frames)
            
            # Filter and store quality frames
            for frame, metrics, idx in zip(batch_frames, batch_metrics, batch_indices):
                if metrics.is_valid and metrics.overall_score >= self.min_quality_threshold:
                    frames.append(frame)
                    quality_metrics.append(metrics)
                    frame_indices.append(idx)
                    
                    if len(frames) >= target_frames:
                        break
            
            await asyncio.sleep(0)
        
        # Calculate quality statistics
        quality_stats = self._calculate_quality_stats(quality_metrics)
        
        return {
            "frames": frames,
            "metrics": quality_metrics,
            "indices": frame_indices
        }, {
            "sampled_frames": frame_count,
            "quality_frames": len(frames),
            "frame_indices": frame_indices,
            "quality_metrics": quality_stats
        }
    
    def _calculate_quality_stats(self, metrics: List[QualityMetrics]) -> Dict:
        """Calculate aggregate quality statistics"""
        if not metrics:
            return {}
            
        return {
            'brightness': np.mean([m.brightness for m in metrics]),
            'contrast': np.mean([m.contrast for m in metrics]),
            'blur_score': np.mean([m.blur_score for m in metrics]),
            'coverage_score': np.mean([m.coverage_score for m in metrics]),
            'overall_score': np.mean([m.overall_score for m in metrics]),
            'quality_variance': np.var([m.overall_score for m in metrics])
        }

    def _create_error_metadata(self, error_message: str) -> Dict:
        """Create metadata for error cases"""
        return {
            "error": error_message,
            "sampled_frames": 0,
            "quality_frames": 0,
            "frame_indices": [],
            "quality_metrics": {}
        }
    
    async def save_upload(self, video: UploadFile) -> Optional[Path]:
        """Save uploaded video file"""
        try:
            # Create temporary file path
            file_extension = video.filename.split('.')[-1]
            temp_file = self.temp_dir / f"temp_{int(time.time())}.{file_extension}"
            
            # Save uploaded file
            with open(temp_file, "wb") as buffer:
                content = await video.read()
                buffer.write(content)
                
            return temp_file
        except Exception as e:
            print(f"Error saving upload: {str(e)}")
            return None

    async def cleanup(self, file_path: Path):
        """Clean up temporary files"""
        try:
            if file_path and file_path.exists():
                file_path.unlink()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")