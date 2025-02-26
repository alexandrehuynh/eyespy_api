from typing import List, Optional, Dict, Tuple
import numpy as np
from ..models import Keypoint

class PoseFusion:
    def __init__(
        self,
        mediapipe_weight: float = 0.6,
        movenet_weight: float = 0.4,
        confidence_threshold: float = 0.3
    ):
        self.mediapipe_weight = mediapipe_weight
        self.movenet_weight = movenet_weight
        self.confidence_threshold = confidence_threshold
        self.keypoint_cache = {}
        self.max_cache_size = 1000

    async def fuse_predictions(
        self,
        mediapipe_keypoints: Optional[List[Keypoint]],
        movenet_keypoints: Optional[List[Keypoint]]
    ) -> Optional[List[Keypoint]]:
        """Fuse predictions using vectorized operations"""
        if not mediapipe_keypoints and not movenet_keypoints:
            return None
            
        if not mediapipe_keypoints:
            return self._adjust_confidence(movenet_keypoints, 0.8)
        if not movenet_keypoints:
            return self._adjust_confidence(mediapipe_keypoints, 0.8)

        # Convert keypoints to numpy arrays for vectorized operations
        mp_array = np.array([
            [kp.x, kp.y, kp.confidence] for kp in mediapipe_keypoints
        ])
        mn_array = np.array([
            [kp.x, kp.y, kp.confidence] for kp in movenet_keypoints
        ])

        # Calculate weighted average using broadcasting
        weights = np.array([self.mediapipe_weight, self.movenet_weight])
        stacked_arrays = np.stack([mp_array, mn_array], axis=-1)
        fused_array = np.sum(stacked_arrays * weights, axis=-1) / np.sum(weights)

        # Create fused keypoints with numpy operations
        fused_keypoints = []
        for i, (pos, mp_kp) in enumerate(zip(fused_array, mediapipe_keypoints)):
            confidence = pos[2]
            if confidence >= self.confidence_threshold:
                fused_keypoints.append(
                    Keypoint(
                        x=pos[0],
                        y=pos[1],
                        confidence=confidence,
                        name=mp_kp.name
                    )
                )

        # Cache results for temporal consistency
        cache_key = hash(tuple(fused_array.flatten()))
        self.keypoint_cache[cache_key] = fused_keypoints
        if len(self.keypoint_cache) > self.max_cache_size:
            self.keypoint_cache.pop(next(iter(self.keypoint_cache)))

        return fused_keypoints if fused_keypoints else None

    def _adjust_confidence(
        self,
        keypoints: List[Keypoint],
        factor: float
    ) -> List[Keypoint]:
        """Adjust confidence scores using vectorized operations"""
        if not keypoints:
            return []
            
        # Convert to numpy array for vectorized operations
        kp_array = np.array([
            [kp.x, kp.y, kp.confidence] for kp in keypoints
        ])
        
        # Adjust confidences
        kp_array[:, 2] *= factor
        
        # Filter and convert back to keypoints
        mask = kp_array[:, 2] >= self.confidence_threshold
        return [
            Keypoint(
                x=pos[0],
                y=pos[1],
                confidence=pos[2],
                name=keypoints[i].name
            )
            for i, pos in enumerate(kp_array[mask])
        ]