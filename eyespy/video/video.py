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

    async def extract_frames(
        self,
        video_path: Path,
        target_fps: Optional[int] = None,
        max_duration: Optional[int] = None
    ) -> AsyncGenerator[Tuple[List[np.ndarray], dict], None]:
        """Extract frames using parallel processing with streaming"""
        if not video_path.exists():
            yield [], self._create_error_metadata("Video file not found")
            return

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            yield [], self._create_error_metadata("Failed to open video file")
            return

        try:
            metadata = await self._get_video_properties(cap)
            if metadata["total_frames"] == 0:
                yield [], self._create_error_metadata("Video appears to be empty")
                return

            metadata.update({
                "target_fps": target_fps or metadata['original_fps'],
                "frame_interval": max(1, metadata["original_fps"] // (target_fps or metadata['original_fps']))
            })

            # Use ThreadPoolExecutor for frame reading
            with ThreadPoolExecutor(max_workers=4) as executor:
                frame_buffer = asyncio.Queue(maxsize=self.buffer_size)
                result_buffer = asyncio.Queue(maxsize=self.buffer_size)
                
                # Start frame reader and quality assessment tasks
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
                        
                        # Process frames in batches
                        frames = result_data['frames']
                        metrics = result_data['metrics']
                        
                        # Filter quality frames using numpy operations
                        quality_mask = np.array([
                            m.is_valid and m.overall_score >= self.min_quality_threshold
                            for m in metrics
                        ])
                        quality_frames = np.array(frames)[quality_mask]
                        
                        if len(quality_frames) > 0:
                            yield quality_frames.tolist(), {
                                **metadata,
                                "frames_processed": len(quality_frames),
                                "quality_metrics": self._calculate_quality_stats(
                                    [m for i, m in enumerate(metrics) if quality_mask[i]]
                                )
                            }
                        
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
            while cap.isOpened():
                ret, frame = await asyncio.get_event_loop().run_in_executor(
                    executor, cap.read
                )
                if not ret:
                    break
                
                if frame_count % metadata['frame_interval'] == 0:
                    await frame_buffer.put({
                        'frame': frame,
                        'index': frame_count
                    })
                
                frame_count += 1
                
            # Signal end of processing
            await frame_buffer.put(None)
            
        except Exception as e:
            logger.error(f"Error in frame reader: {str(e)}")
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
        """Process frames for quality with proper async handling"""
        try:
            while True:
                batch_data = await frame_buffer.get()
                if batch_data is None:
                    break
                
                # Process batch with proper async handling
                batch_metrics = await self._process_batch(batch_data['frames'])
                
                # Add quality results to result buffer
                await result_buffer.put({
                    'frames': batch_data['frames'],
                    'indices': batch_data['indices'],
                    'metrics': batch_metrics
                })
                
        finally:
            result_buffer.put(None)

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
        """Get video properties with optimized settings"""
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        # Target settings for processing efficiency
        target_fps = min(30, fps)  # Don't exceed original FPS
        target_height = 720  # Resize for processing efficiency
        target_width = int(width * (target_height / height))

        # Calculate frame sampling to maintain smooth motion
        sample_interval = max(1, int(fps / target_fps))

        return {
            'original_fps': fps,
            'total_frames': total_frames,
            'original_dimensions': (height, width),
            'target_dimensions': (target_height, target_width),
            'duration': duration,
            'sample_interval': sample_interval,
            'target_fps': target_fps,
            'resize_factor': target_height / height
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