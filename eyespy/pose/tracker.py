import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import asyncio
from ..utils.executor_service import get_executor
from ..utils.async_utils import run_in_executor

@dataclass
class TrackedKeypoint:
    name: str
    position: Tuple[float, float]
    confidence: float
    velocity: Tuple[float, float]
    history: List[Tuple[float, float]]
    confidence_history: List[float]

class ConfidenceTracker:
    def __init__(
        self,
        history_size: int = 5,
        smoothing_factor: float = 0.3,
        max_displacement: float = 0.2,
        confidence_decay: float = 0.9
    ):
        self.history_size = history_size
        self.smoothing_factor = smoothing_factor
        self.max_displacement = max_displacement
        self.confidence_decay = confidence_decay
        self.tracked_keypoints: Dict[str, TrackedKeypoint] = {}

    async def update(
        self,
        keypoints: Dict[str, Tuple[float, float, float]]
    ) -> Dict[str, Tuple[float, float, float]]:
        """Update tracking with new keypoints"""
        # Process each keypoint in parallel
        tasks = []

        for name, (x, y, confidence) in keypoints.items():
            task = self._process_keypoint_async(name, (x, y), confidence)
            tasks.append(task)

        # Wait for all keypoint processing to complete
        processed_keypoints = await asyncio.gather(*tasks)

        # Update tracked keypoints with processed results
        return {
            name: (x, y, conf) 
            for name, (x, y, conf) in zip(keypoints.keys(), processed_keypoints)
        }

    async def _process_keypoint_async(
        self,
        name: str,
        position: Tuple[float, float],
        confidence: float
    ) -> Tuple[float, float, float]:
        """Process a single keypoint asynchronously"""
        return await run_in_executor(
            self._process_keypoint,
            name,
            position,
            confidence
        )

    def _process_keypoint(
        self,
        name: str,
        position: Tuple[float, float],
        confidence: float
    ) -> Tuple[float, float, float]:
        """Process a single keypoint"""
        if name not in self.tracked_keypoints:
            # Initialize new tracked keypoint
            self.tracked_keypoints[name] = TrackedKeypoint(
                name=name,
                position=position,
                confidence=confidence,
                velocity=(0.0, 0.0),
                history=[position],
                confidence_history=[confidence]
            )
            return (*position, confidence)

        tracked = self.tracked_keypoints[name]
        
        # Calculate displacement and velocity
        displacement = self._calculate_displacement(tracked.position, position)
        
        # Check if movement is valid
        if displacement > self.max_displacement:
            # Movement too large, likely erroneous
            confidence *= 0.5
            position = self._estimate_position(tracked)
        
        # Update velocity
        new_velocity = (
            position[0] - tracked.position[0],
            position[1] - tracked.position[1]
        )
        
        # Smooth velocity
        tracked.velocity = (
            tracked.velocity[0] * (1 - self.smoothing_factor) + new_velocity[0] * self.smoothing_factor,
            tracked.velocity[1] * (1 - self.smoothing_factor) + new_velocity[1] * self.smoothing_factor
        )
        
        # Update history
        tracked.history.append(position)
        tracked.confidence_history.append(confidence)
        if len(tracked.history) > self.history_size:
            tracked.history.pop(0)
            tracked.confidence_history.pop(0)
        
        # Smooth position and confidence
        smoothed_position = self._smooth_position(tracked)
        smoothed_confidence = self._smooth_confidence(tracked)
        
        # Update tracked keypoint
        tracked.position = smoothed_position
        tracked.confidence = smoothed_confidence
        
        return (*smoothed_position, smoothed_confidence)

    def _calculate_displacement(
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float]
    ) -> float:
        """Calculate displacement between two positions"""
        return np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)

    def _estimate_position(self, tracked: TrackedKeypoint) -> Tuple[float, float]:
        """Estimate position based on history and velocity"""
        if len(tracked.history) < 2:
            return tracked.position
            
        # Use velocity-based prediction
        predicted_x = tracked.position[0] + tracked.velocity[0]
        predicted_y = tracked.position[1] + tracked.velocity[1]
        
        return (predicted_x, predicted_y)

    def _smooth_position(self, tracked: TrackedKeypoint) -> Tuple[float, float]:
        """Apply exponential smoothing to position"""
        if len(tracked.history) < 2:
            return tracked.position
            
        weights = np.exp([i * self.smoothing_factor for i in range(len(tracked.history))])
        weights = weights / np.sum(weights)
        
        x = sum(p[0] * w for p, w in zip(tracked.history, weights))
        y = sum(p[1] * w for p, w in zip(tracked.history, weights))
        
        return (x, y)

    def _smooth_confidence(self, tracked: TrackedKeypoint) -> float:
        """Smooth confidence values over time"""
        if len(tracked.confidence_history) < 2:
            return tracked.confidence
            
        # Apply temporal decay
        weights = [self.confidence_decay ** i for i in range(len(tracked.confidence_history))]
        weights = [w / sum(weights) for w in weights]
        
        return sum(c * w for c, w in zip(tracked.confidence_history, weights))

    def reset(self):
        """Reset tracker state"""
        self.tracked_keypoints.clear()