from fastapi import UploadFile
import time
import cv2
import numpy as np
import asyncio
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import deque
from ..config import settings
from .quality import AdaptiveFrameQualityAssessor, QualityMetrics
from .frame_selector import FrameSelector

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
        if not self.frames and not self.completed.is_set():
            await self.not_empty.wait()
        
        async with self.lock:
            if self.frames:
                return self.frames.popleft()
            return None

    def mark_completed(self):
        self.completed.set()

class VideoProcessor:
    def __init__(self):
        self.temp_dir = Path(settings.TEMP_DIR)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.quality_assessor = AdaptiveFrameQualityAssessor()
        self.frame_selector = FrameSelector()
        self.batch_size = 30
        self.buffer_size = 100

    async def extract_frames(
        self,
        video_path: Path,
        target_fps: int = 30,
        max_duration: int = 10
    ) -> Tuple[List[np.ndarray], dict]:
        """Extract frames using parallel processing"""
        cap = cv2.VideoCapture(str(video_path))
        
        try:
            # Calibration phase
            calibration_data = await self._perform_calibration(cap)
            if not calibration_data:
                return [], self._create_error_metadata("Calibration failed")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            metadata = self._get_video_properties(cap)
            
            # Create shared buffer
            frame_buffer = FrameBuffer(maxsize=self.buffer_size)
            result_buffer = FrameBuffer(maxsize=self.buffer_size)
            
            # Start parallel processing tasks
            tasks = [
                asyncio.create_task(self._frame_reader(cap, frame_buffer, metadata)),
                asyncio.create_task(self._quality_assessor(frame_buffer, result_buffer)),
                asyncio.create_task(self._frame_selector(result_buffer))
            ]
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for exceptions
            for result in results:
                if isinstance(result, Exception):
                    raise result
            
            # Process results
            selected_frames, selection_stats = results[2]  # Results from frame selector
            
            metadata.update(selection_stats)
            return selected_frames, metadata
            
        except Exception as e:
            return [], self._create_error_metadata(str(e))
        finally:
            cap.release()

    async def _frame_reader(
        self,
        cap: cv2.VideoCapture,
        frame_buffer: FrameBuffer,
        metadata: Dict
    ):
        """Read frames and add to buffer"""
        frame_count = 0
        try:
            while cap.isOpened():
                batch_frames = []
                batch_indices = []
                
                # Read batch of frames
                for _ in range(self.batch_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    batch_frames.append(frame)
                    batch_indices.append(frame_count)
                    frame_count += 1
                
                if not batch_frames:
                    break
                
                # Add batch to buffer
                await frame_buffer.put({
                    'frames': batch_frames,
                    'indices': batch_indices
                })
                
                await asyncio.sleep(0)
                
        finally:
            frame_buffer.mark_completed()

    async def _quality_assessor(
        self,
        frame_buffer: FrameBuffer,
        result_buffer: FrameBuffer
    ):
        """Process frames for quality in parallel"""
        try:
            while True:
                batch_data = await frame_buffer.get()
                if batch_data is None:
                    break
                
                # Process batch in parallel
                batch_metrics = await self._process_batch(batch_data['frames'])
                
                # Add quality results to result buffer
                await result_buffer.put({
                    'frames': batch_data['frames'],
                    'indices': batch_data['indices'],
                    'metrics': batch_metrics
                })
                
        finally:
            result_buffer.mark_completed()

    async def _frame_selector(
        self,
        result_buffer: FrameBuffer
    ) -> Tuple[List[np.ndarray], Dict]:
        """Select frames based on quality and distribution"""
        frames = []
        indices = []
        metrics = []
        
        try:
            while True:
                result_data = await result_buffer.get()
                if result_data is None:
                    break
                
                # Add to collection for selection
                frames.extend(result_data['frames'])
                indices.extend(result_data['indices'])
                metrics.extend(result_data['metrics'])
            
            # Perform final selection
            selected_frames, selected_indices, selection_stats = (
                self.frame_selector.select_best_frames(
                    frames,
                    metrics
                )
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

    async def _perform_calibration(self, cap: cv2.VideoCapture) -> Optional[Dict]:
        """Perform initial calibration and return calibration data"""
        try:
            calibration_frames = []
            while len(calibration_frames) < self.quality_assessor.calibration_size:
                ret, frame = cap.read()
                if not ret:
                    break
                calibration_frames.append(frame)
                
                if len(calibration_frames) % 5 == 0:
                    await asyncio.sleep(0)
            
            if calibration_frames:
                self.quality_assessor.calibrate_thresholds(calibration_frames)
                return {'success': True, 'frames_sampled': len(calibration_frames)}
            
            return None
            
        except Exception as e:
            print(f"Calibration error: {str(e)}")
            return None

    def _get_video_properties(self, cap: cv2.VideoCapture) -> Dict:
        """Get video properties and calculate sampling parameters"""
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return {
            "original_fps": fps,
            "total_frames": total_frames,
            "dimensions": (width, height),
            "duration": total_frames / fps if fps else 0,
            "sample_interval": max(1, total_frames // self.max_frames_to_process)
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
                if frame_count >= self.max_frames_to_process:
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

    async def _process_batch(self, frames: List[np.ndarray]) -> List[QualityMetrics]:
        """Process a batch of frames in parallel"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: [self.quality_assessor.assess_frame(frame) for frame in frames]
        )

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