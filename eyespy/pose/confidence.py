from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

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
            self.core_points = {
                "LEFT_SHOULDER", "RIGHT_SHOULDER",
                "LEFT_HIP", "RIGHT_HIP"
            }
        
        # Primary limb points (upper arms and legs)
        if self.primary_limb_points is None:
            self.primary_limb_points = {
                "LEFT_ELBOW", "RIGHT_ELBOW",
                "LEFT_KNEE", "RIGHT_KNEE"
            }
        
        # Secondary limb points (lower arms and legs)
        if self.secondary_limb_points is None:
            self.secondary_limb_points = {
                "LEFT_WRIST", "RIGHT_WRIST",
                "LEFT_ANKLE", "RIGHT_ANKLE"
            }
            
        # Face points
        if self.face_points is None:
            self.face_points = {
                "NOSE", "LEFT_EYE", "RIGHT_EYE",
                "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT"
            }
        
        # Extremity points (fingers and toes)
        if self.extremity_points is None:
            self.extremity_points = {
                "LEFT_PINKY", "RIGHT_PINKY",
                "LEFT_INDEX", "RIGHT_INDEX",
                "LEFT_THUMB", "RIGHT_THUMB",
                "LEFT_HEEL", "RIGHT_HEEL",
                "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
            }

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
    def __init__(self, thresholds: Optional[ConfidenceThresholds] = None):
        self.thresholds = thresholds or ConfidenceThresholds()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Define keypoint relationships for parallel processing
        self.keypoint_groups = {
            'core': ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP'],
            'arms': ['LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST'],
            'legs': ['LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE'],
            'face': ['NOSE', 'LEFT_EYE', 'RIGHT_EYE', 'LEFT_EAR', 'RIGHT_EAR']
        }

    async def assess_keypoints(
        self,
        keypoints: Dict[str, tuple],
        confidences: Dict[str, float]
    ) -> Dict[str, float]:
        """Assess keypoint confidences in parallel"""
        loop = asyncio.get_event_loop()
        
        # Create tasks for parallel anatomical checks
        tasks = [
            loop.run_in_executor(
                self.executor,
                self._check_group_consistency,
                group_name,
                keypoint_list,
                keypoints,
                confidences
            )
            for group_name, keypoint_list in self.keypoint_groups.items()
        ]
        
        # Add parallel confidence adjustment tasks
        tasks.extend([
            loop.run_in_executor(
                self.executor,
                self._adjust_confidence,
                name,
                pos,
                conf,
                keypoints,
                confidences
            )
            for name, (pos, conf) in zip(keypoints.keys(), zip(keypoints.values(), confidences.values()))
        ])
        
        # Wait for all assessments to complete
        results = await asyncio.gather(*tasks)
        
        # Combine group consistency checks and adjusted confidences
        group_results = results[:len(self.keypoint_groups)]
        confidence_results = results[len(self.keypoint_groups):]
        
        # Merge results
        final_confidences = {}
        for name, adjusted_conf in zip(keypoints.keys(), confidence_results):
            group_factor = self._get_group_factor(name, group_results)
            final_confidences[name] = min(1.0, adjusted_conf * group_factor)
            
        return final_confidences

    def _check_group_consistency(
        self,
        group_name: str,
        keypoint_list: List[str],
        keypoints: Dict[str, tuple],
        confidences: Dict[str, float]
    ) -> Tuple[str, float]:
        """Check anatomical consistency within a group"""
        if group_name == 'core':
            return self._check_core_consistency(keypoints, confidences)
        elif group_name == 'arms':
            return self._check_arm_consistency(keypoints, confidences)
        elif group_name == 'legs':
            return self._check_leg_consistency(keypoints, confidences)
        elif group_name == 'face':
            return self._check_face_consistency(keypoints, confidences)
        return group_name, 1.0

    def _check_core_consistency(
        self,
        keypoints: Dict[str, tuple],
        confidences: Dict[str, float]
    ) -> Tuple[str, float]:
        """Check core body consistency"""
        if all(k in keypoints for k in ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP']):
            ls = keypoints['LEFT_SHOULDER']
            rs = keypoints['RIGHT_SHOULDER']
            lh = keypoints['LEFT_HIP']
            rh = keypoints['RIGHT_HIP']
            
            # Check if shoulders are above hips
            if ls[1] < lh[1] and rs[1] < rh[1]:
                return 'core', 1.0
        return 'core', 0.8

    def _check_arm_consistency(
        self,
        keypoints: Dict[str, tuple],
        confidences: Dict[str, float]
    ) -> Tuple[str, float]:
        """Check arm consistency in parallel"""
        score = 1.0
        for side in ['LEFT', 'RIGHT']:
            if all(f'{side}_{part}' in keypoints for part in ['SHOULDER', 'ELBOW', 'WRIST']):
                s = keypoints[f'{side}_SHOULDER']
                e = keypoints[f'{side}_ELBOW']
                w = keypoints[f'{side}_WRIST']
                
                # Check if elbow is between shoulder and wrist
                if min(s[1], w[1]) <= e[1] <= max(s[1], w[1]):
                    score *= 1.0
                else:
                    score *= 0.8
        return 'arms', score

    def _check_leg_consistency(
        self,
        keypoints: Dict[str, tuple],
        confidences: Dict[str, float]
    ) -> Tuple[str, float]:
        """Check leg consistency in parallel"""
        score = 1.0
        for side in ['LEFT', 'RIGHT']:
            if all(f'{side}_{part}' in keypoints for part in ['HIP', 'KNEE', 'ANKLE']):
                h = keypoints[f'{side}_HIP']
                k = keypoints[f'{side}_KNEE']
                a = keypoints[f'{side}_ANKLE']
                
                # Check if knee is between hip and ankle
                if h[1] < k[1] < a[1]:
                    score *= 1.0
                else:
                    score *= 0.8
        return 'legs', score

    def _check_face_consistency(
        self,
        keypoints: Dict[str, tuple],
        confidences: Dict[str, float]
    ) -> Tuple[str, float]:
        """Check face landmark consistency"""
        if 'NOSE' in keypoints:
            nose = keypoints['NOSE']
            score = 1.0
            
            # Check eye positions relative to nose
            for side in ['LEFT', 'RIGHT']:
                if f'{side}_EYE' in keypoints:
                    eye = keypoints[f'{side}_EYE']
                    if eye[1] - nose[1] > 0.1:  # Eyes should be near nose height
                        score *= 0.8
            return 'face', score
        return 'face', 0.8

    def _adjust_confidence(
        self,
        keypoint_name: str,
        position: tuple,
        confidence: float,
        keypoints: Dict[str, tuple],
        confidences: Dict[str, float]
    ) -> float:
        """Adjust individual keypoint confidence"""
        base_threshold = self.thresholds.get_threshold(keypoint_name)
        
        # Apply position-based adjustments
        if keypoint_name in self.keypoint_groups['core']:
            return min(1.0, confidence * 1.2)  # Boost core keypoints
        elif keypoint_name in self.keypoint_groups['extremities']:
            return min(1.0, confidence * 0.9)  # Reduce extremity confidence
            
        return confidence

    def _get_group_factor(self, keypoint_name: str, group_results: List[Tuple[str, float]]) -> float:
        """Get group consistency factor for a keypoint"""
        for group_name, score in group_results:
            if any(group_list for group_list in self.keypoint_groups.values() if keypoint_name in group_list):
                return score
        return 1.0

    async def process_batch(
        self,
        batch_keypoints: List[Dict[str, tuple]],
        batch_confidences: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """Process multiple keypoint sets in parallel"""
        return await asyncio.gather(*[
            self.assess_keypoints(kps, confs)
            for kps, confs in zip(batch_keypoints, batch_confidences)
        ])

    def __del__(self):
        """Cleanup executor"""
        self.executor.shutdown(wait=False)