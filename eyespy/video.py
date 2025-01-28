# app/video.py
import cv2
import numpy as np
from pathlib import Path
import asyncio
from typing import List, Tuple, Dict, Optional
from .config import settings
from .video.quality import AdaptiveFrameQualityAssessor, QualityMetrics
from .video.frame_selector import FrameSelector

class VideoProcessor:
    def __init__(self):
        self.temp_dir = Path(settings.TEMP_DIR)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.quality_assessor = AdaptiveFrameQualityAssessor()
        self.frame_selector = FrameSelector()
        
        # Dynamic parameters for adaptive processing
        self.batch_size = 30
        self.min_quality_threshold = 0.4
        self.max_frames_to_process = 1000

    async def extract_frames(
        self,
        video_path: Path,
        target_fps: int = 30,
        max_duration: int = 10
    ) -> Tuple[List[np.ndarray], dict]:
        """Extract frames with both quality assessment and frame selection"""
        cap = cv2.VideoCapture(str(video_path))
        
        try:
            # Phase 1: Initial calibration
            calibration_data = await self._perform_calibration(cap)
            if not calibration_data:
                return [], self._create_error_metadata("Calibration failed")
            
            # Reset video capture after calibration
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Get video properties
            metadata = self._get_video_properties(cap)
            target_frames = min(target_fps * max_duration, metadata["total_frames"])
            
            # Phase 2: Main frame extraction with quality assessment
            candidate_frames, quality_data = await self._process_frames(
                cap,
                target_frames,
                metadata["sample_interval"]
            )
            
            # Phase 3: Frame selection
            selected_frames, selected_indices, selection_stats = (
                self.frame_selector.select_best_frames(
                    candidate_frames["frames"],
                    candidate_frames["metrics"],
                    target_count=target_fps
                )
            )
            
            # Update metadata with both quality and selection information
            metadata.update(quality_data)
            metadata.update({
                "selected_frames": len(selected_frames),
                "selected_indices": selected_indices,
                "selection_stats": selection_stats
            })
            
            return selected_frames, metadata
            
        except Exception as e:
            return [], self._create_error_metadata(str(e))
        finally:
            cap.release()

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