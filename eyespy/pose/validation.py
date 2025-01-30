from typing import List, Dict, Optional, Tuple
import numpy as np
from ..models import Keypoint

class PoseValidator:
    def __init__(self):
        # Define joint pairs for angle calculations
        self.joint_pairs = {
            "elbow_right": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
            "elbow_left": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
            "knee_right": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
            "knee_left": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
            "neck": ("LEFT_SHOULDER", "NOSE", "RIGHT_SHOULDER"),
            "hips": ("LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE")
        }
        
        # Define anatomical constraints
        self.angle_constraints = {
            "elbow_right": (0, 180),  # Elbow can't bend backwards
            "elbow_left": (0, 180),
            "knee_right": (0, 180),   # Knee can't bend backwards
            "knee_left": (0, 180),
            "neck": (45, 135),        # Head shouldn't rotate too extremely
            "hips": (45, 135)         # Hips shouldn't twist too extremely
        }
        
        # Define expected relative positions
        self.position_rules = [
            # (point1, point2, dimension, expected_relation)
            ("LEFT_SHOULDER", "LEFT_ELBOW", "y", "greater"),  # Shoulder above elbow
            ("RIGHT_SHOULDER", "RIGHT_ELBOW", "y", "greater"),
            ("LEFT_HIP", "LEFT_KNEE", "y", "greater"),       # Hip above knee
            ("RIGHT_HIP", "RIGHT_KNEE", "y", "greater"),
            ("LEFT_KNEE", "LEFT_ANKLE", "y", "greater"),     # Knee above ankle
            ("RIGHT_KNEE", "RIGHT_ANKLE", "y", "greater")
        ]

    def calculate_angle(
        self,
        point1: Keypoint,
        point2: Keypoint,
        point3: Keypoint
    ) -> float:
        """Calculate angle between three points"""
        try:
            # Convert to numpy arrays for easier calculation
            p1 = np.array([point1.x, point1.y])
            p2 = np.array([point2.x, point2.y])
            p3 = np.array([point3.x, point3.y])
            
            # Calculate vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            
            return angle
        except:
            return 0.0

    def check_angle_constraints(
        self,
        keypoints: List[Keypoint]
    ) -> Dict[str, bool]:
        """Check if joint angles are within anatomical constraints"""
        keypoint_dict = {kp.name: kp for kp in keypoints}
        angle_validations = {}
        
        for joint_name, (p1_name, p2_name, p3_name) in self.joint_pairs.items():
            # Check if we have all required points
            if all(name in keypoint_dict for name in (p1_name, p2_name, p3_name)):
                p1 = keypoint_dict[p1_name]
                p2 = keypoint_dict[p2_name]
                p3 = keypoint_dict[p3_name]
                
                angle = self.calculate_angle(p1, p2, p3)
                min_angle, max_angle = self.angle_constraints[joint_name]
                
                angle_validations[joint_name] = min_angle <= angle <= max_angle
            else:
                angle_validations[joint_name] = False
                
        return angle_validations

    def check_position_rules(
        self,
        keypoints: List[Keypoint]
    ) -> Dict[str, bool]:
        """Check if keypoints follow expected relative positions"""
        keypoint_dict = {kp.name: kp for kp in keypoints}
        position_validations = {}
        
        for p1_name, p2_name, dim, relation in self.position_rules:
            rule_name = f"{p1_name}_{p2_name}_{dim}"
            
            if p1_name in keypoint_dict and p2_name in keypoint_dict:
                p1 = keypoint_dict[p1_name]
                p2 = keypoint_dict[p2_name]
                
                val1 = getattr(p1, dim)
                val2 = getattr(p2, dim)
                
                if relation == "greater":
                    position_validations[rule_name] = val1 < val2
                else:
                    position_validations[rule_name] = val1 > val2
            else:
                position_validations[rule_name] = False
                
        return position_validations

    def check_symmetry(self, keypoints: List[Keypoint]) -> Dict[str, float]:
        """Vectorized symmetry check"""
        keypoint_dict = {kp.name: kp for kp in keypoints}
        symmetry_scores = {}
        
        symmetry_pairs = [
            ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
            ("LEFT_HIP", "RIGHT_HIP"),
            ("LEFT_KNEE", "RIGHT_KNEE"),
            ("LEFT_ANKLE", "RIGHT_ANKLE")
        ]
        
        # Vectorized calculation
        for left, right in symmetry_pairs:
            if left in keypoint_dict and right in keypoint_dict:
                left_x = keypoint_dict[left].x
                right_x = keypoint_dict[right].x
                
                # Calculate using numpy vectorization
                x_diff = np.abs(np.array([left_x, right_x]) - 0.5)
                symmetry_score = 1.0 - np.mean(np.abs(x_diff[0] - x_diff[1]))
                symmetry_scores[f"{left}_{right}"] = symmetry_score
                
        return symmetry_scores

    def validate_pose(
        self,
        keypoints: List[Keypoint]
    ) -> Tuple[bool, Dict[str, any]]:
        """Complete pose validation with detailed metrics"""
        # Run all validation checks
        angle_checks = self.check_angle_constraints(keypoints)
        position_checks = self.check_position_rules(keypoints)
        symmetry_scores = self.check_symmetry(keypoints)
        
        # Calculate overall validation score
        angle_score = sum(angle_checks.values()) / len(angle_checks)
        position_score = sum(position_checks.values()) / len(position_checks)
        symmetry_score = sum(symmetry_scores.values()) / len(symmetry_scores) if symmetry_scores else 0
        
        overall_score = (angle_score + position_score + symmetry_score) / 3
        
        # Determine if pose is valid (you can adjust the threshold)
        is_valid = overall_score >= 0.7
        
        validation_metrics = {
            "is_valid": is_valid,
            "overall_score": overall_score,
            "angle_validations": angle_checks,
            "position_validations": position_checks,
            "symmetry_scores": symmetry_scores,
            "component_scores": {
                "angle_score": angle_score,
                "position_score": position_score,
                "symmetry_score": symmetry_score
            }
        }
        
        return is_valid, validation_metrics