# app/pose/confidence.py
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class ConfidenceThresholds:
    # Core body (crucial points that should be very reliable)
    core_threshold: float = 0.65    # Reduced from 0.7 to account for real-world conditions
    core_points: List[str] = None
    
    # Primary limbs (important but can be slightly less reliable)
    primary_limb_threshold: float = 0.55
    primary_limb_points: List[str] = None
    
    # Secondary limbs (can have lower confidence)
    secondary_limb_threshold: float = 0.45
    secondary_limb_points: List[str] = None
    
    # Extremities (naturally more difficult to detect)
    extremity_threshold: float = 0.35    # Increased from 0.3 for better reliability
    extremity_points: List[str] = None
    
    # Face points (can vary based on angle)
    face_threshold: float = 0.40
    face_points: List[str] = None
    
    # Global fallback
    global_threshold: float = 0.45    # Baseline for unlisted points

    def __post_init__(self):
        # Core body points (torso)
        if self.core_points is None:
            self.core_points = [
                "LEFT_SHOULDER", "RIGHT_SHOULDER",
                "LEFT_HIP", "RIGHT_HIP"
            ]
        
        # Primary limb points (upper arms and legs)
        if self.primary_limb_points is None:
            self.primary_limb_points = [
                "LEFT_ELBOW", "RIGHT_ELBOW",
                "LEFT_KNEE", "RIGHT_KNEE"
            ]
        
        # Secondary limb points (lower arms and legs)
        if self.secondary_limb_points is None:
            self.secondary_limb_points = [
                "LEFT_WRIST", "RIGHT_WRIST",
                "LEFT_ANKLE", "RIGHT_ANKLE"
            ]
            
        # Face points
        if self.face_points is None:
            self.face_points = [
                "NOSE", "LEFT_EYE", "RIGHT_EYE",
                "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT"
            ]
        
        # Extremity points (fingers and toes)
        if self.extremity_points is None:
            self.extremity_points = [
                "LEFT_PINKY", "RIGHT_PINKY",
                "LEFT_INDEX", "RIGHT_INDEX",
                "LEFT_THUMB", "RIGHT_THUMB",
                "LEFT_HEEL", "RIGHT_HEEL",
                "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
            ]

    def get_threshold(self, keypoint_name: str) -> float:
        """Get confidence threshold for a specific keypoint"""
        if keypoint_name in self.core_points:
            return self.core_threshold
        elif keypoint_name in self.primary_limb_points:
            return self.primary_limb_threshold
        elif keypoint_name in self.secondary_limb_points:
            return self.secondary_limb_threshold
        elif keypoint_name in self.face_points:
            return self.face_threshold
        elif keypoint_name in self.extremity_points:
            return self.extremity_threshold
        return self.global_threshold

    def get_confidence_category(self, keypoint_name: str) -> str:
        """Get the category of a keypoint for confidence assessment"""
        if keypoint_name in self.core_points:
            return "core"
        elif keypoint_name in self.primary_limb_points:
            return "primary_limb"
        elif keypoint_name in self.secondary_limb_points:
            return "secondary_limb"
        elif keypoint_name in self.face_points:
            return "face"
        elif keypoint_name in self.extremity_points:
            return "extremity"
        return "other"

class ThresholdAdjuster:
    """Utility class to help adjust thresholds based on detection statistics"""
    
    def __init__(self, thresholds: ConfidenceThresholds):
        self.thresholds = thresholds
        self.detection_stats = {
            "core": [],
            "primary_limb": [],
            "secondary_limb": [],
            "face": [],
            "extremity": [],
            "other": []
        }

    def record_detection(self, keypoint_name: str, confidence: float):
        """Record a detection for statistical analysis"""
        category = self.thresholds.get_confidence_category(keypoint_name)
        self.detection_stats[category].append(confidence)

    def analyze_detection_rates(self) -> Dict[str, Dict[str, float]]:
        """Analyze detection statistics for each category"""
        analysis = {}
        
        for category, confidences in self.detection_stats.items():
            if confidences:
                analysis[category] = {
                    "mean_confidence": np.mean(confidences),
                    "median_confidence": np.median(confidences),
                    "std_confidence": np.std(confidences),
                    "detection_rate": len([c for c in confidences if c > 0]) / len(confidences),
                    "sample_size": len(confidences)
                }
        
        return analysis

class AdaptiveConfidenceAssessor:
    """Enhanced confidence assessor with adaptive thresholds"""
    
    def __init__(self, thresholds: Optional[ConfidenceThresholds] = None):
        self.thresholds = thresholds or ConfidenceThresholds()
        self.adjuster = ThresholdAdjuster(self.thresholds)
        
        # Confidence adjustment factors
        self.motion_penalty = 0.1       # Reduce confidence for fast motion
        self.stability_bonus = 0.15     # Boost confidence for stable detections
        self.context_bonus = 0.2        # Boost for good surrounding detections

    def assess_confidence(
        self,
        keypoint_name: str,
        raw_confidence: float,
        is_stable: bool = True,
        has_motion: bool = False,
        context_score: float = 0.5
    ) -> float:
        """
        Assess and adjust confidence based on multiple factors
        """
        base_threshold = self.thresholds.get_threshold(keypoint_name)
        
        # Start with raw confidence
        adjusted_confidence = raw_confidence
        
        # Apply stability bonus
        if is_stable:
            adjusted_confidence = min(1.0, adjusted_confidence + self.stability_bonus)
        
        # Apply motion penalty
        if has_motion:
            adjusted_confidence *= (1.0 - self.motion_penalty)
        
        # Apply context bonus if surrounding detections are good
        if context_score > 0.7:
            adjusted_confidence = min(1.0, adjusted_confidence + self.context_bonus)
        
        # Record for statistical analysis
        self.adjuster.record_detection(keypoint_name, adjusted_confidence)
        
        return adjusted_confidence

    def get_detection_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get detection statistics for analysis"""
        return self.adjuster.analyze_detection_rates()