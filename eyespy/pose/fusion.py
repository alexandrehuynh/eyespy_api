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

    async def fuse_predictions(
        self,
        mediapipe_keypoints: Optional[List[Keypoint]],
        movenet_keypoints: Optional[List[Keypoint]]
    ) -> Optional[List[Keypoint]]:
        """Fuse predictions from both models"""
        if not mediapipe_keypoints and not movenet_keypoints:
            return None
            
        # If one model fails, use the other with reduced confidence
        if not mediapipe_keypoints:
            return self._adjust_confidence(movenet_keypoints, 0.8)
        if not movenet_keypoints:
            return self._adjust_confidence(mediapipe_keypoints, 0.8)
        
        # Create keypoint dictionaries for easy lookup
        mp_dict = {kp.name: kp for kp in mediapipe_keypoints}
        mn_dict = {kp.name: kp for kp in movenet_keypoints}
        
        # Fuse keypoints
        fused_keypoints = []
        
        # Process all unique keypoint names
        for name in set(mp_dict.keys()) | set(mn_dict.keys()):
            mp_kp = mp_dict.get(name)
            mn_kp = mn_dict.get(name)
            
            if mp_kp and mn_kp:
                # Both models detected the keypoint
                fused_kp = self._fuse_keypoint(mp_kp, mn_kp)
                if fused_kp.confidence >= self.confidence_threshold:
                    fused_keypoints.append(fused_kp)
            elif mp_kp:
                # Only MediaPipe detected the keypoint
                if mp_kp.confidence >= self.confidence_threshold:
                    fused_keypoints.append(mp_kp)
            elif mn_kp:
                # Only MoveNet detected the keypoint
                if mn_kp.confidence >= self.confidence_threshold:
                    fused_keypoints.append(mn_kp)
        
        return fused_keypoints if fused_keypoints else None

    def _fuse_keypoint(
        self,
        mp_keypoint: Keypoint,
        mn_keypoint: Keypoint
    ) -> Keypoint:
        """Fuse individual keypoint predictions"""
        # Calculate weighted coordinates
        x = (
            mp_keypoint.x * self.mediapipe_weight * mp_keypoint.confidence +
            mn_keypoint.x * self.movenet_weight * mn_keypoint.confidence
        ) / (
            self.mediapipe_weight * mp_keypoint.confidence +
            self.movenet_weight * mn_keypoint.confidence
        )
        
        y = (
            mp_keypoint.y * self.mediapipe_weight * mp_keypoint.confidence +
            mn_keypoint.y * self.movenet_weight * mn_keypoint.confidence
        ) / (
            self.mediapipe_weight * mp_keypoint.confidence +
            self.movenet_weight * mn_keypoint.confidence
        )
        
        # Combine confidences
        confidence = max(
            mp_keypoint.confidence * self.mediapipe_weight,
            mn_keypoint.confidence * self.movenet_weight
        )
        
        return Keypoint(
            x=x,
            y=y,
            confidence=confidence,
            name=mp_keypoint.name
        )

    def _adjust_confidence(
        self,
        keypoints: List[Keypoint],
        factor: float
    ) -> List[Keypoint]:
        """Adjust confidence values by a factor"""
        return [
            Keypoint(
                x=kp.x,
                y=kp.y,
                confidence=kp.confidence * factor,
                name=kp.name
            )
            for kp in keypoints
        ]