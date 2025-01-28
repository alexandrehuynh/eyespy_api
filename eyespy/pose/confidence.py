from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class ConfidenceThresholds:
    # Core body (higher threshold as these are crucial)
    core_threshold: float = 0.7
    core_points: List[str] = None
    
    # Limbs (medium threshold)
    limb_threshold: float = 0.5
    limb_points: List[str] = None
    
    # Extremities (lower threshold as these are harder to detect)
    extremity_threshold: float = 0.3
    extremity_points: List[str] = None
    
    # Global fallback
    global_threshold: float = 0.5

    def __post_init__(self):
        # Define keypoint groups if not provided
        if self.core_points is None:
            self.core_points = [
                "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
                "LEFT_HIP", "RIGHT_HIP"
            ]
        
        if self.limb_points is None:
            self.limb_points = [
                "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_KNEE", "RIGHT_KNEE",
                "LEFT_EAR", "RIGHT_EAR", "LEFT_EYE", "RIGHT_EYE"
            ]
        
        if self.extremity_points is None:
            self.extremity_points = [
                "LEFT_WRIST", "RIGHT_WRIST", "LEFT_ANKLE", "RIGHT_ANKLE",
                "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX",
                "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HEEL", "RIGHT_HEEL",
                "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
            ]

    def get_threshold(self, keypoint_name: str) -> float:
        """Get confidence threshold for a specific keypoint"""
        if keypoint_name in self.core_points:
            return self.core_threshold
        elif keypoint_name in self.limb_points:
            return self.limb_threshold
        elif keypoint_name in self.extremity_points:
            return self.extremity_threshold
        return self.global_threshold

class ConfidenceAssessor:
    def __init__(self, thresholds: Optional[ConfidenceThresholds] = None):
        self.thresholds = thresholds or ConfidenceThresholds()
        
        # Keypoint relationships for contextual confidence
        self.keypoint_relationships = {
            "LEFT_WRIST": ["LEFT_ELBOW"],
            "RIGHT_WRIST": ["RIGHT_ELBOW"],
            "LEFT_ANKLE": ["LEFT_KNEE"],
            "RIGHT_ANKLE": ["RIGHT_KNEE"],
            "LEFT_ELBOW": ["LEFT_SHOULDER"],
            "RIGHT_ELBOW": ["RIGHT_SHOULDER"],
            "LEFT_KNEE": ["LEFT_HIP"],
            "RIGHT_KNEE": ["RIGHT_HIP"]
        }

    def adjust_confidence(self, keypoint_name: str, confidence: float, nearby_confidences: Dict[str, float]) -> float:
        """Adjust confidence based on context"""
        # Get related keypoints
        related_points = self.keypoint_relationships.get(keypoint_name, [])
        
        if not related_points:
            return confidence
            
        # Calculate average confidence of related points
        related_confidences = [
            nearby_confidences.get(point, 0.0)
            for point in related_points
        ]
        avg_related_confidence = np.mean(related_confidences) if related_confidences else 0.0
        
        # Adjust confidence based on context
        # If nearby points have high confidence, slightly boost this point's confidence
        if avg_related_confidence > 0.8:
            return min(1.0, confidence * 1.2)
        # If nearby points have very low confidence, reduce this point's confidence
        elif avg_related_confidence < 0.3:
            return confidence * 0.8
            
        return confidence

    def check_anatomical_consistency(self, keypoints: Dict[str, tuple]) -> Dict[str, float]:
        """Check if keypoint positions make anatomical sense"""
        consistency_scores = {}
        
        # Check shoulder-hip relationships
        if all(k in keypoints for k in ["LEFT_SHOULDER", "LEFT_HIP", "RIGHT_SHOULDER", "RIGHT_HIP"]):
            ls, lh = keypoints["LEFT_SHOULDER"], keypoints["LEFT_HIP"]
            rs, rh = keypoints["RIGHT_SHOULDER"], keypoints["RIGHT_HIP"]
            
            # Shoulders should be above hips
            shoulder_hip_score = 1.0 if (ls[1] < lh[1] and rs[1] < rh[1]) else 0.5
            consistency_scores.update({
                "LEFT_SHOULDER": shoulder_hip_score,
                "RIGHT_SHOULDER": shoulder_hip_score,
                "LEFT_HIP": shoulder_hip_score,
                "RIGHT_HIP": shoulder_hip_score
            })
        
        return consistency_scores

    def assess_keypoint(
        self,
        keypoint_name: str,
        confidence: float,
        position: tuple,
        all_keypoints: Dict[str, tuple],
        all_confidences: Dict[str, float]
    ) -> Tuple[float, bool]:
        """Assess a single keypoint's confidence with context"""
        # Get base threshold
        threshold = self.thresholds.get_threshold(keypoint_name)
        
        # Adjust confidence based on nearby keypoints
        adjusted_confidence = self.adjust_confidence(
            keypoint_name,
            confidence,
            all_confidences
        )
        
        # Get anatomical consistency score
        consistency_scores = self.check_anatomical_consistency(all_keypoints)
        consistency_factor = consistency_scores.get(keypoint_name, 1.0)
        
        # Final confidence score
        final_confidence = adjusted_confidence * consistency_factor
        
        # Determine if keypoint is valid
        is_valid = final_confidence >= threshold
        
        return final_confidence, is_valid
