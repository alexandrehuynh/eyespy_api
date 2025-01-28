from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from .quality import QualityMetrics

@dataclass
class FrameScore:
    frame_index: int
    quality_score: float
    blur_score: float
    brightness_score: float
    coverage_score: float
    overall_score: float

class FrameSelector:
    def __init__(
        self,
        target_fps: int = 30,
        min_quality_score: float = 0.5,
        max_blur_score: float = 0.7,
        selection_window: int = 5
    ):
        self.target_fps = target_fps
        self.min_quality_score = min_quality_score
        self.max_blur_score = max_blur_score
        self.selection_window = selection_window

    def calculate_frame_score(
        self,
        quality_metrics: QualityMetrics,
        frame_index: int
    ) -> FrameScore:
        """Calculate comprehensive score for frame"""
        # Weighted quality metrics
        weights = {
            'quality': 0.4,
            'blur': 0.3,
            'brightness': 0.2,
            'coverage': 0.1
        }

        overall_score = (
            quality_metrics.overall_score * weights['quality'] +
            quality_metrics.blur_score * weights['blur'] +
            quality_metrics.brightness * weights['brightness'] +
            quality_metrics.coverage_score * weights['coverage']
        )

        return FrameScore(
            frame_index=frame_index,
            quality_score=quality_metrics.overall_score,
            blur_score=quality_metrics.blur_score,
            brightness_score=quality_metrics.brightness,
            coverage_score=quality_metrics.coverage_score,
            overall_score=overall_score
        )

    def select_best_frames(
        self,
        frames: List[np.ndarray],
        quality_metrics: List[QualityMetrics],
        target_count: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[int], Dict[str, float]]:
        """Select the best frames based on quality metrics"""
        if not frames or not quality_metrics:
            return [], [], {}

        # Calculate scores for all frames
        frame_scores = [
            self.calculate_frame_score(metrics, idx)
            for idx, metrics in enumerate(quality_metrics)
        ]

        # Filter frames that meet minimum quality criteria
        valid_scores = [
            score for score in frame_scores
            if (score.quality_score >= self.min_quality_score and
                score.blur_score <= self.max_blur_score)
        ]

        if not valid_scores:
            return [], [], {}

        # Determine target number of frames if not specified
        if target_count is None:
            target_count = len(frames) // self.selection_window

        # Select best frames using sliding window
        selected_indices = []
        current_idx = 0
        window_size = len(frames) // target_count if target_count > 0 else 1

        while len(selected_indices) < target_count and current_idx < len(frames):
            # Get scores in current window
            window_end = min(current_idx + window_size, len(frame_scores))
            window_scores = frame_scores[current_idx:window_end]
            
            if window_scores:
                # Select frame with highest score in window
                best_score = max(window_scores, key=lambda x: x.overall_score)
                selected_indices.append(best_score.frame_index)
            
            current_idx += window_size

        # Get selected frames and their indices
        selected_frames = [frames[i] for i in selected_indices]
        
        # Calculate selection statistics
        stats = {
            'selected_frames': len(selected_frames),
            'average_quality': np.mean([frame_scores[i].quality_score for i in selected_indices]),
            'average_blur': np.mean([frame_scores[i].blur_score for i in selected_indices]),
            'average_brightness': np.mean([frame_scores[i].brightness_score for i in selected_indices]),
            'average_coverage': np.mean([frame_scores[i].coverage_score for i in selected_indices]),
            'average_overall': np.mean([frame_scores[i].overall_score for i in selected_indices])
        }

        return selected_frames, selected_indices, stats
