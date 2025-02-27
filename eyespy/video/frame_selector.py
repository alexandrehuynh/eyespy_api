import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import asyncio
from .quality import QualityMetrics
from ..utils.executor_service import get_executor
from ..utils.async_utils import run_in_executor, gather_with_concurrency

@dataclass
class FrameScore:
    frame_index: int
    quality_score: float
    blur_score: float
    brightness_score: float
    coverage_score: float
    temporal_score: float
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
        self.batch_size = 50

    async def select_best_frames(
            self,
            frames: List[np.ndarray],
            quality_metrics: List[QualityMetrics],
            target_count: Optional[int] = None
        ) -> Tuple[List[np.ndarray], List[int], Dict[str, float]]:
            """Select best frames using parallel processing"""
            if not frames or not quality_metrics:
                return [], [], {}

            if target_count is None:
                target_count = len(frames) // self.selection_window

            # Calculate batch sizes for parallel processing
            total_frames = len(frames)
            batch_size = min(self.batch_size, max(10, total_frames // 4))
            
            # Process batches in parallel with concurrency limit
            batch_tasks = []
            
            for i in range(0, total_frames, batch_size):
                batch_frames = frames[i:i + batch_size]
                batch_metrics = quality_metrics[i:i + batch_size]
                
                task = self._score_batch(batch_frames, batch_metrics, i)
                batch_tasks.append(task)

            # Use concurrency control to limit parallel execution
            batch_results = await gather_with_concurrency(4, *batch_tasks)
            
            # Combine all scores
            all_scores = []
            for batch_scores in batch_results:
                all_scores.extend(batch_scores)

            # Select best frames
            selected_indices = await self._select_distributed_frames(
                all_scores,
                target_count,
                len(frames)
            )

            # Gather selected frames and statistics
            selected_frames = [frames[i] for i in selected_indices]
            selection_stats = await self._calculate_selection_stats(
                selected_indices,
                all_scores,
                quality_metrics
            )

            return selected_frames, selected_indices, selection_stats

    async def _score_batch(
        self,
        batch_frames: List[np.ndarray],
        batch_metrics: List[QualityMetrics],
        start_idx: int
    ) -> List[FrameScore]:
        """Score a batch of frames"""
        return await run_in_executor(
            self._score_batch_sync,
            batch_frames,
            batch_metrics,
            start_idx
        )
    
    def _score_batch_sync(
        self,
        batch_frames: List[np.ndarray],
        batch_metrics: List[QualityMetrics],
        start_idx: int
    ) -> List[FrameScore]:
        """Synchronous implementation of batch scoring for executor"""
        scores = []
        total_frames = len(batch_frames)
        
        for i, (frame, metrics) in enumerate(zip(batch_frames, batch_metrics)):
            frame_idx = start_idx + i
            
            # Calculate temporal score based on frame position
            temporal_score = self._calculate_temporal_score(
                frame_idx,
                total_frames
            )
            
            # Combine scores
            overall_score = self._calculate_overall_score(
                metrics,
                temporal_score
            )
            
            scores.append(FrameScore(
                frame_index=frame_idx,
                quality_score=metrics.overall_score,
                blur_score=metrics.blur_score,
                brightness_score=metrics.brightness,
                coverage_score=metrics.coverage_score,
                temporal_score=temporal_score,
                overall_score=overall_score
            ))
        
        return scores

    def _calculate_temporal_score(self, frame_idx: int, total_frames: int) -> float:
        """Calculate temporal distribution score"""
        # Prefer frames that are evenly distributed
        ideal_spacing = total_frames / self.target_fps
        position_score = 1.0 - (frame_idx % ideal_spacing) / ideal_spacing
        return position_score

    def _calculate_overall_score(
        self,
        metrics: QualityMetrics,
        temporal_score: float
    ) -> float:
        """Calculate weighted overall score"""
        weights = {
            'quality': 0.4,
            'blur': 0.2,
            'brightness': 0.1,
            'coverage': 0.1,
            'temporal': 0.2
        }
        
        return (
            metrics.overall_score * weights['quality'] +
            metrics.blur_score * weights['blur'] +
            metrics.brightness * weights['brightness'] +
            metrics.coverage_score * weights['coverage'] +
            temporal_score * weights['temporal']
        )

    async def _select_distributed_frames(
        self,
        scores: List[FrameScore],
        target_count: int,
        total_frames: int
    ) -> List[int]:
        """Select frames with parallel window processing"""
        if not scores:
            return []

        # Sort scores by overall score
        sorted_scores = sorted(scores, key=lambda x: x.overall_score, reverse=True)
        
        # Precompute ideal spacing once
        ideal_spacing = total_frames / target_count
        frame_indices = np.arange(total_frames)
        position_scores = 1.0 - (frame_indices % ideal_spacing) / ideal_spacing
        
        # Add precomputed scores to sorted list
        for score in sorted_scores:
            score.temporal_score = position_scores[score.frame_index]

        # Calculate window size for temporal distribution
        window_size = total_frames // target_count

        # Process selection windows in parallel with concurrency control
        window_tasks = []
        
        for i in range(0, total_frames, window_size):
            window_end = min(i + window_size, total_frames)
            window_scores = [
                s for s in sorted_scores
                if i <= s.frame_index < window_end
            ]
            
            if window_scores:
                task = run_in_executor(
                    self._select_best_in_window,
                    window_scores
                )
                window_tasks.append(task)

        # Use concurrency control to limit parallel execution
        selected_from_windows = await gather_with_concurrency(8, *window_tasks)
        
        # Combine and sort final selections
        selected_indices = []
        for indices in selected_from_windows:
            if indices:
                selected_indices.extend(indices)

        return sorted(selected_indices)[:target_count]

    def _select_best_in_window(self, window_scores: List[FrameScore]) -> List[int]:
        """Select best frame in a temporal window"""
        if not window_scores:
            return []
        
        # Select frame with highest score in window
        best_score = max(window_scores, key=lambda x: x.overall_score)
        return [best_score.frame_index]

    async def _calculate_selection_stats(
        self,
        selected_indices: List[int],
        scores: List[FrameScore],
        quality_metrics: List[QualityMetrics]
    ) -> Dict[str, float]:
        """Calculate statistics for selected frames"""
        return await run_in_executor(
            self._calculate_selection_stats_sync,
            selected_indices,
            scores,
            quality_metrics
        )
    
    def _calculate_selection_stats_sync(
        self,
        selected_indices: List[int],
        scores: List[FrameScore],
        quality_metrics: List[QualityMetrics]
    ) -> Dict[str, float]:
        """Synchronous implementation of statistics calculation for executor"""
        selected_scores = [
            s for s in scores if s.frame_index in selected_indices
        ]
        
        if not selected_scores:
            return {}

        return {
            'average_quality': np.mean([s.quality_score for s in selected_scores]),
            'average_blur': np.mean([s.blur_score for s in selected_scores]),
            'average_brightness': np.mean([s.brightness_score for s in selected_scores]),
            'average_coverage': np.mean([s.coverage_score for s in selected_scores]),
            'temporal_distribution': np.std(selected_indices) / len(scores),
            'selection_rate': len(selected_indices) / len(scores)
        }