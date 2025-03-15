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
from ..utils.executor_service import get_executor
from ..utils.async_utils import run_in_executor, gather_with_concurrency
import psutil
import concurrent.futures

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MotionThresholdDetector:
    """Detect significant motion in video to optimize frame selection"""
    def __init__(self, threshold: float = 50.0, min_motion_area: float = 0.01):
        self.threshold = threshold
        self.min_motion_area = min_motion_area  # Minimum percentage of frame that must change
        self.last_frame = None
        self.motion_history = []
        
    def reset(self):
        """Reset the detector state"""
        self.last_frame = None
        self.motion_history = []
        
    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if there's significant motion between this frame and the last
        
        Args:
            frame: Current video frame
            
        Returns:
            (has_motion, motion_score) tuple
        """
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # If this is the first frame, save it and return no motion
        if self.last_frame is None:
            self.last_frame = gray
            self.motion_history.append(0.0)
            return False, 0.0
            
        # Calculate absolute difference
        frame_diff = cv2.absdiff(self.last_frame, gray)
        
        # Threshold the difference
        _, thresholded = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of pixels that changed
        motion_pixels = np.count_nonzero(thresholded)
        total_pixels = frame.shape[0] * frame.shape[1]
        motion_percent = motion_pixels / total_pixels
        
        # Determine if motion is significant
        has_motion = motion_percent > self.min_motion_area
        
        # Update last frame
        self.last_frame = gray
        
        # Update motion history
        self.motion_history.append(motion_percent * 100)  # Store as percentage
        if len(self.motion_history) > 30:  # Keep last 30 frames
            self.motion_history.pop(0)
            
        return has_motion, motion_percent * 100  # Return as percentage

class AdaptiveFrameSelector:
    """Intelligently select frames for processing based on quality and motion"""
    def __init__(
        self, 
        target_fps: float = 15.0,
        min_quality_threshold: float = 0.5,
        motion_threshold: float = 50.0
    ):
        self.target_fps = target_fps
        self.min_quality_threshold = min_quality_threshold
        self.quality_assessor = AdaptiveFrameQualityAssessor()
        self.motion_detector = MotionThresholdDetector(threshold=motion_threshold)
        
        # Statistics
        self.frames_read = 0
        self.frames_selected = 0
        self.frames_with_motion = 0
        self.avg_quality = 0.0
        
    async def select_frames(
        self, 
        video_path: Path,
        max_frames: Optional[int] = None
    ) -> AsyncGenerator[Tuple[np.ndarray, Dict], None]:
        """
        Extract and select high-quality frames with motion awareness
        
        This intelligent frame selector ensures we:
        1. Extract frames at approximately the target FPS
        2. Prioritize frames with significant motion
        3. Maintain only high quality frames
        4. Adapt to video content dynamically
        """
        # Reset state
        self.frames_read = 0
        self.frames_selected = 0
        self.frames_with_motion = 0
        self.avg_quality = 0.0
        self.motion_detector.reset()
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return
            
        try:
            # Get video properties
            orig_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / orig_fps if orig_fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate minimum frames to skip based on FPS ratio
            min_frame_interval = max(1, int(orig_fps / self.target_fps))
            
            # Metadata for frames
            video_info = {
                "fps": orig_fps,
                "width": width,
                "height": height,
                "total_frames": total_frames,
                "duration": duration,
                "target_fps": self.target_fps,
                "min_frame_interval": min_frame_interval
            }
            
            logger.info(f"Processing video: {orig_fps} FPS, {width}x{height}, {duration:.1f}s")
            logger.info(f"Target FPS: {self.target_fps}, min frame interval: {min_frame_interval}")
            
            # Frame accumulator for adaptive selection
            frame_buffer = []
            frame_qualities = []
            frame_timestamps = []
            
            # Dynamic frame interval - will adapt based on content
            dynamic_interval = min_frame_interval
            
            # Process frames
            frame_idx = 0
            last_selected_idx = -100  # Force first frame to be selected
            
            while cap.isOpened():
                ret, frame = await run_in_executor(cap.read)
                if not ret:
                    break
                    
                self.frames_read += 1
                frame_idx += 1
                
                # Force yield at regular minimal intervals, but also check for motion detection
                should_process = False
                
                # Always process the first frame
                if frame_idx == 1:
                    should_process = True
                    
                # Check if we've reached the minimal interval
                elif frame_idx - last_selected_idx >= dynamic_interval:
                    should_process = True
                    
                if should_process:
                    # Check for significant motion
                    has_motion, motion_score = self.motion_detector.detect_motion(frame)
                    if has_motion:
                        self.frames_with_motion += 1
                    
                    # Assess quality
                    quality = await self.quality_assessor.assess_frame(frame)
                    
                    # For high motion or high quality, select the frame
                    if quality.is_valid and quality.overall_score >= self.min_quality_threshold:
                        # Mark this as the last selected frame
                        last_selected_idx = frame_idx
                        
                        # Update statistics
                        self.frames_selected += 1
                        self.avg_quality = ((self.avg_quality * (self.frames_selected - 1)) + 
                                            quality.overall_score) / self.frames_selected
                        
                        # Create metadata for this frame
                        frame_metadata = {
                            **video_info,
                            "frame_index": frame_idx,
                            "timestamp": frame_idx / orig_fps,
                            "quality_score": quality.overall_score,
                            "motion_score": motion_score,
                            "has_motion": has_motion,
                            "frames_read": self.frames_read,
                            "frames_selected": self.frames_selected,
                            "selection_ratio": self.frames_selected / self.frames_read,
                            "avg_quality": self.avg_quality
                        }
                        
                        # Dynamically adjust interval based on motion
                        if has_motion:
                            # If there's motion, reduce interval to capture more frames
                            dynamic_interval = max(1, min_frame_interval - 1)
                        else:
                            # If no motion, increase interval to skip more frames
                            dynamic_interval = min_frame_interval + 1
                        
                        # Yield the selected frame
                        yield frame, frame_metadata
                        
                        # Check if we've reached the limit
                        if max_frames and self.frames_selected >= max_frames:
                            logger.info(f"Reached max frames limit: {max_frames}")
                            break
                
                # Give control back to event loop periodically
                if frame_idx % 10 == 0:
                    await asyncio.sleep(0)
            
            # Log final statistics
            logger.info(f"Frame selection complete: read {self.frames_read}, selected {self.frames_selected}")
            logger.info(f"Selection ratio: {self.frames_selected / self.frames_read:.2f}, average quality: {self.avg_quality:.3f}")
            logger.info(f"Frames with motion: {self.frames_with_motion} ({self.frames_with_motion / self.frames_selected:.1%} of selected)")
            
        finally:
            cap.release()

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
        min_quality_threshold: float = 0.5,
        process_every_n_frames: int = 1,
        adaptive_batching: bool = True,
        use_motion_detection: bool = True
    ):
        """Initialize VideoProcessor with configurable parameters"""
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_quality_threshold = min_quality_threshold
        self.process_every_n_frames = process_every_n_frames
        self.adaptive_batching = adaptive_batching
        self.use_motion_detection = use_motion_detection
        self.temp_dir = Path(settings.TEMP_DIR)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.quality_assessor = AdaptiveFrameQualityAssessor()
        self.active_tasks = set()  # Track active tasks
        
        # Adaptive settings - avoid using event loop properties directly
        # Instead, use psutil to determine a reasonable number of workers
        cpu_count = psutil.cpu_count(logical=False) or 4
        self.max_concurrent_tasks = min(8, cpu_count)
        
        # New adaptive frame selector
        self.frame_selector = AdaptiveFrameSelector(
            min_quality_threshold=min_quality_threshold
        )
        
        # Performance metrics
        self.frame_processing_times = []
        self.quality_assessment_times = []
        
        logger.info(f"VideoProcessor initialized with {self.max_concurrent_tasks} concurrent tasks")
        logger.info(f"Adaptive batching: {self.adaptive_batching}, Process every N frames: {self.process_every_n_frames}")
        logger.info(f"Motion detection: {self.use_motion_detection}")

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
        """Extract frames with performance optimization and adaptive frame skipping"""
        if not video_path.exists():
            yield [], self._create_error_metadata("Video file not found")
            return

        # Use either basic extraction or advanced motion-aware extraction
        if self.use_motion_detection:
            # Initialize the adaptive frame selector with the target FPS
            actual_target_fps = target_fps or 15.0
            self.frame_selector.target_fps = actual_target_fps
            
            # Calculate max frames based on max_duration if provided
            max_frames = None
            if max_duration:
                max_frames = int(max_duration * actual_target_fps)
                
            # Use the advanced frame selector
            batch_frames = []
            batch_metadata = {}
            metadata_set = False
            
            async for frame, frame_metadata in self.frame_selector.select_frames(
                video_path, max_frames=max_frames
            ):
                # Keep metadata from first frame
                if not metadata_set:
                    batch_metadata = frame_metadata
                    metadata_set = True
                
                # Add frame to batch
                batch_frames.append(frame)
                
                # When we reach batch size, yield the batch
                if len(batch_frames) >= self.batch_size:
                    # Update metadata
                    batch_metadata["frames_processed"] = len(batch_frames)
                    
                    # Yield the batch
                    yield batch_frames, batch_metadata
                    
                    # Clear batch
                    batch_frames = []
            
            # Yield any remaining frames
            if batch_frames:
                batch_metadata["frames_processed"] = len(batch_frames)
                yield batch_frames, batch_metadata
                
        else:
            # Use the original frame extraction method
            cap = cv2.VideoCapture(str(video_path))
            try:
                metadata = await self._get_video_properties(cap)
                original_fps = metadata.get('fps', 30)
                
                # Calculate appropriate frame interval
                frame_interval = 1
                if target_fps and target_fps < original_fps:
                    frame_interval = max(1, round(original_fps / target_fps))
                frame_interval *= self.process_every_n_frames  # Apply additional skipping
                
                logger.info(f"Video properties: {metadata.get('width')}x{metadata.get('height')} @ {original_fps} FPS")
                logger.info(f"Processing with frame interval: {frame_interval} (1 frame every {frame_interval} frames)")
    
                # Determine optimal batch size based on system resources
                memory_info = psutil.virtual_memory()
                cpu_count = psutil.cpu_count(logical=False) or 4
                
                # Adaptive batch size calculation
                if self.adaptive_batching:
                    # Base size on available memory and CPU cores
                    # Assuming each 1080p frame is about 6MB in memory
                    frame_size_estimate = metadata.get('width', 1920) * metadata.get('height', 1080) * 3 / (1024 * 1024)
                    available_memory_mb = memory_info.available / (1024 * 1024)
                    
                    # Target using at most 25% of available memory
                    memory_based_batch = int(available_memory_mb * 0.25 / frame_size_estimate)
                    
                    # Combine with CPU-based batch size (more CPUs = larger batches)
                    dynamic_batch_size = min(memory_based_batch, self.batch_size * cpu_count // 2)
                    
                    # Keep within reasonable bounds
                    adaptive_batch_size = max(10, min(dynamic_batch_size, 100))
                    logger.info(f"Using adaptive batch size: {adaptive_batch_size} frames based on system resources")
                    self.batch_size = adaptive_batch_size
                
                # Frame processing
                frame_count = 0
                batch_frames = []
                batch_timestamps = []
                
                # Continuous processing
                while cap.isOpened():
                    # Use shared executor to read frames
                    ret, frame = await run_in_executor(cap.read)
                    if not ret:
                        break
                    
                    # Only process frames based on calculated interval
                    if frame_count % frame_interval == 0:
                        batch_frames.append(frame)
                        batch_timestamps.append(frame_count / original_fps)
                        
                        # When batch is full, process and yield
                        if len(batch_frames) >= self.batch_size:
                            # Pre-filter frames using parallel processing for quality assessment
                            quality_assessment_start = time.time()
                            quality_assessment_tasks = [
                                self.quality_assessor.assess_frame(frm)
                                for frm in batch_frames
                            ]
                            
                            # Use gather with concurrency to limit parallel processing
                            batch_metrics = await gather_with_concurrency(
                                self.max_concurrent_tasks,
                                *quality_assessment_tasks
                            )
                            
                            self.quality_assessment_times.append(time.time() - quality_assessment_start)
                            
                            # Filter quality frames and their timestamps together
                            quality_frames = []
                            quality_timestamps = []
                            
                            for i, (frm, metric) in enumerate(zip(batch_frames, batch_metrics)):
                                if metric.is_valid and metric.overall_score >= self.min_quality_threshold:
                                    quality_frames.append(frm)
                                    quality_timestamps.append(batch_timestamps[i])
                            
                            # Yield only quality frames
                            if quality_frames:
                                # Update metadata with frame timestamps
                                batch_metadata = {
                                    **metadata,
                                    "frames_processed": len(quality_frames),
                                    "frame_timestamps": quality_timestamps,
                                    "original_frame_count": frame_count,
                                    "average_quality_score": sum(m.overall_score for m in batch_metrics) / len(batch_metrics) if batch_metrics else 0,
                                }
                                
                                yield quality_frames, batch_metadata
                                
                            # Clear the batch
                            batch_frames = []
                            batch_timestamps = []
                    
                    frame_count += 1
                    
                    # Give control back to other tasks periodically
                    if frame_count % 10 == 0:
                        await asyncio.sleep(0)
                
                # Process any remaining frames
                if batch_frames:
                    # Process remaining frames in parallel
                    quality_assessment_tasks = [
                        self.quality_assessor.assess_frame(frm)
                        for frm in batch_frames
                    ]
                    
                    batch_metrics = await gather_with_concurrency(
                        self.max_concurrent_tasks,
                        *quality_assessment_tasks
                    )
                    
                    # Filter quality frames with their timestamps
                    quality_frames = []
                    quality_timestamps = []
                    
                    for i, (frm, metric) in enumerate(zip(batch_frames, batch_metrics)):
                        if metric.is_valid and metric.overall_score >= self.min_quality_threshold:
                            quality_frames.append(frm)
                            quality_timestamps.append(batch_timestamps[i])
                    
                    if quality_frames:
                        batch_metadata = {
                            **metadata,
                            "frames_processed": len(quality_frames),
                            "frame_timestamps": quality_timestamps,
                            "original_frame_count": frame_count,
                            "average_quality_score": sum(m.overall_score for m in batch_metrics) / len(batch_metrics) if batch_metrics else 0,
                        }
                        
                        yield quality_frames, batch_metadata
                
            finally:
                cap.release()
            
    async def _frame_reader(
        self,
        cap: cv2.VideoCapture,
        frame_buffer: asyncio.Queue,
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
                # Use shared executor to read frames
                ret, frame = await run_in_executor(cap.read)
                
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
        # Process frames in parallel with concurrency control
        return await gather_with_concurrency(
            4,  # Process up to 4 frames at a time
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
                logger.warning("No frames collected for selection")
                return [], {}

            # Use the frame selector with the shared executor
            selected_frames, selected_indices, selection_stats = await self.frame_selector.select_best_frames(
                frames,
                metrics
            )
            
            stats = {
                'quality_stats': await run_in_executor(self._calculate_quality_stats, metrics),
                'selection_stats': selection_stats,
                'frame_indices': selected_indices
            }
            
            return selected_frames, stats
            
        except Exception as e:
            logger.error(f"Frame selection error: {str(e)}")
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
                # Use shared executor for frame reading
                ret, frame = await run_in_executor(cap.read)
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
            
            logger.info(f"Collected {len(calibration_frames)} frames for calibration")
            
            if calibration_frames:
                if await self.quality_assessor.calibrate_thresholds(calibration_frames):
                    return {
                        'success': True,
                        'frames_sampled': len(calibration_frames),
                        'frame_count': frame_count
                    }
            
            logger.warning("Failed to collect enough calibration frames")
            return None
            
        except Exception as e:
            logger.error(f"Calibration error: {str(e)}")
            return None

    async def _get_video_properties(self, cap: cv2.VideoCapture) -> Dict:
        """Get video properties with frame interval calculation"""
        # Run in executor to avoid blocking the event loop
        return await run_in_executor(
            self._get_video_properties_sync,
            cap
        )
    
    def _get_video_properties_sync(self, cap: cv2.VideoCapture) -> Dict:
        """Synchronous implementation of getting video properties"""
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
            
            # Read batch of frames using shared executor
            for _ in range(self.batch_size):
                if frame_count >= self.batch_size:
                    break
                    
                ret, frame = await run_in_executor(cap.read)
                if not ret:
                    break
                
                if frame_count % sample_interval == 0:
                    batch_frames.append(frame)
                    batch_indices.append(frame_count)
                
                frame_count += 1
            
            if not batch_frames:
                break
            
            # Process batch with concurrency control
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
        
        return {
            "frames": frames,
            "metrics": quality_metrics,
            "indices": frame_indices
        }, {
            "sampled_frames": frame_count,
            "quality_frames": len(frames),
            "frame_indices": frame_indices,
            "quality_metrics": self._calculate_quality_stats(quality_metrics)
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
            logger.error(f"Error saving upload: {str(e)}")
            return None

    async def cleanup(self, file_path: Path):
        """Clean up temporary files"""
        try:
            if file_path and file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")