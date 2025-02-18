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
    def __init__(self):
        self.temp_dir = Path(settings.TEMP_DIR)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.quality_assessor = AdaptiveFrameQualityAssessor()
        self.frame_selector = FrameSelector()
        self.batch_size = 30
        self.buffer_size = 100
        self.max_frames_to_process = 300 
        self.min_quality_threshold = 0.3 

    async def extract_frames(
        self,
        video_path: Path,
        target_fps: int = 30,
        max_duration: int = 10
    ) -> AsyncGenerator[Tuple[List[np.ndarray], dict], None]:
        """Extract frames using parallel processing with streaming"""
        if not video_path.exists():
            print(f"Video file not found: {video_path}")
            yield [], self._create_error_metadata("Video file not found")
            return

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            yield [], self._create_error_metadata("Failed to open video file")
            return

        try:
            # Get video properties first
            metadata = self._get_video_properties(cap)
            print(f"Video properties: {metadata}")

            if metadata["total_frames"] == 0:
                yield [], self._create_error_metadata("Video appears to be empty")
                return

            # Calculate frame limits
            max_frames = min(
                self.max_frames_to_process,
                int(target_fps * max_duration),
                metadata["total_frames"]
            )
            
            # Update metadata
            metadata.update({
                "max_frames": max_frames,
                "target_fps": target_fps,
                "frame_interval": max(1, metadata["original_fps"] // target_fps)
            })

            # Calibration phase
            print("Starting calibration...")
            calibration_data = await self._perform_calibration(cap)
            if not calibration_data:
                print("Calibration failed")
                yield [], self._create_error_metadata("Calibration failed")
                return
            
            print("Calibration completed successfully")
            metadata["calibration"] = calibration_data
            
            # Reset video position
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Process frames in chunks
            async for frames_chunk, chunk_metadata in self._process_frame_chunks(cap, metadata):
                if frames_chunk:
                    yield frames_chunk, {**metadata, **chunk_metadata}
                    
        except Exception as e:
            print(f"Error in extract_frames: {str(e)}")
            yield [], self._create_error_metadata(str(e))
        finally:
            cap.release()

    async def _process_frame_chunks(
        self,
        cap: cv2.VideoCapture,
        metadata: Dict
    ) -> AsyncGenerator[Tuple[List[np.ndarray], Dict], None]:
        """Process video frames in chunks"""
        frame_buffer = FrameBuffer(maxsize=self.buffer_size)
        result_buffer = FrameBuffer(maxsize=self.buffer_size)
        
        try:
            # Start processing tasks
            reader_task = asyncio.create_task(
                self._frame_reader(cap, frame_buffer, metadata)
            )
            quality_task = asyncio.create_task(
                self._quality_assessor(frame_buffer, result_buffer)
            )
            
            frames_processed = 0
            current_chunk: List[np.ndarray] = []
            chunk_metadata: Dict = {"chunk_index": 0}
            
            while True:
                result_data = await result_buffer.get()
                if result_data is None:  # End of processing
                    if current_chunk:  # Yield final chunk
                        yield current_chunk, chunk_metadata
                    break
                
                # Process quality results
                quality_frames = [
                    frame for frame, metrics in zip(
                        result_data['frames'], 
                        result_data['metrics']
                    )
                    if metrics.is_valid and metrics.overall_score >= self.min_quality_threshold
                ]
                
                current_chunk.extend(quality_frames)
                frames_processed += len(quality_frames)
                
                # Yield chunk when it reaches batch size
                if len(current_chunk) >= self.batch_size:
                    chunk_metadata.update({
                        "frames_processed": frames_processed,
                        "chunk_size": len(current_chunk),
                        "memory_usage": self._get_memory_usage()
                    })
                    
                    yield current_chunk, chunk_metadata
                    
                    # Reset chunk
                    current_chunk = []
                    chunk_metadata = {
                        "chunk_index": chunk_metadata["chunk_index"] + 1
                    }
                
                await asyncio.sleep(0)  # Allow other tasks to run
                
        except Exception as e:
            print(f"Error in frame chunk processing: {str(e)}")
            yield [], {"error": str(e)}
        finally:
            # Cleanup tasks
            for task in [reader_task, quality_task]:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Clear buffers
            await self._clear_buffer(frame_buffer)
            await self._clear_buffer(result_buffer)

    async def _frame_reader(
        self,
        cap: cv2.VideoCapture,
        frame_buffer: FrameBuffer,
        metadata: Dict
    ):
        """Read frames and add to buffer"""
        try:
            frames_read = 0
            frame_interval = metadata["frame_interval"]
            max_frames = metadata["max_frames"]
            
            while frames_read < max_frames:
                batch_frames = []
                batch_indices = []
                
                # Read batch of frames
                for _ in range(min(self.batch_size, max_frames - frames_read)):
                    # Skip frames according to target FPS
                    for _ in range(frame_interval - 1):
                        cap.grab()
                    
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = self._optimize_frame(frame)
                    batch_frames.append(frame)
                    batch_indices.append(frames_read)
                    frames_read += 1
                
                if not batch_frames:
                    break
                
                await frame_buffer.put({
                    'frames': batch_frames,
                    'indices': batch_indices
                })
                
                await asyncio.sleep(0)
                
        finally:
            frame_buffer.mark_completed()

    async def _process_batch(self, frames: List[np.ndarray]) -> List[QualityMetrics]:
        """Process a batch of frames with proper async handling"""
        return await asyncio.gather(
            *[self.quality_assessor.assess_frame(frame) for frame in frames]
        )

    async def _frame_selector(self, result_buffer: FrameBuffer) -> Tuple[List[np.ndarray], Dict]:
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

    async def _quality_assessor(self, frame_buffer: FrameBuffer, result_buffer: FrameBuffer):
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
            result_buffer.mark_completed()

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