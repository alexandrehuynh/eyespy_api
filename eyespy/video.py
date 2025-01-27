import cv2
import numpy as np
from pathlib import Path
import asyncio
from typing import List, Tuple, Optional
from .config import settings
from .video.quality import FrameQualityAssessor, QualityMetrics

class VideoProcessor:
    def __init__(self):
        self.temp_dir = Path(settings.TEMP_DIR)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.quality_assessor = FrameQualityAssessor()

    async def extract_frames(
        self,
        video_path: Path,
        target_fps: int = 30,
        max_duration: int = 10
    ) -> Tuple[List[np.ndarray], dict]:
        """Extract frames with quality assessment"""
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame sampling
        target_frames = min(target_fps * max_duration, total_frames)
        sample_interval = max(1, total_frames // target_frames)
        
        frames = []
        frame_indices = []
        quality_metrics = []
        frame_count = 0
        
        try:
            while cap.isOpened() and len(frames) < target_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_interval == 0:
                    # Assess frame quality
                    quality = self.quality_assessor.assess_frame(frame)
                    
                    if quality.is_valid:
                        frames.append(frame)
                        frame_indices.append(frame_count)
                        quality_metrics.append(quality)
                
                frame_count += 1
                
                # Allow other async operations
                if frame_count % 10 == 0:
                    await asyncio.sleep(0)
            
            # Calculate average quality metrics
            avg_quality = {
                'brightness': np.mean([q.brightness for q in quality_metrics]),
                'contrast': np.mean([q.contrast for q in quality_metrics]),
                'blur_score': np.mean([q.blur_score for q in quality_metrics]),
                'coverage_score': np.mean([q.coverage_score for q in quality_metrics]),
                'overall_score': np.mean([q.overall_score for q in quality_metrics])
            }
            
            metadata = {
                "original_fps": original_fps,
                "processed_fps": target_fps,
                "total_frames": total_frames,
                "sampled_frames": len(frames),
                "quality_frames": len(frames),
                "duration": duration,
                "frame_indices": frame_indices,
                "dimensions": (width, height),
                "quality_metrics": avg_quality
            }
            
            return frames, metadata
            
        finally:
            cap.release()