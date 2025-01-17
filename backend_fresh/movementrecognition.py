import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

@dataclass
class MovementPhase:
    name: str
    joint_angles: Dict[str, Tuple[float, float]]  # (min, max) angles for each joint
    duration_range: Tuple[int, int]  # (min, max) frames
    required_sequence: List[str]  # Required preceding phases

@dataclass
class MovementPattern:
    name: str
    phases: List[MovementPhase]
    bilateral: bool = True  # Whether movement should be symmetric

class MovementRecognitionSystem:
    def __init__(self):
        self.movement_patterns = {
            'squat': MovementPattern(
                name='squat',
                phases=[
                    MovementPhase(
                        name='descent',
                        joint_angles={
                            'left_knee': (90, 130),  # Knee flexion range
                            'right_knee': (90, 130),
                            'left_hip': (70, 110),   # Hip flexion range
                            'right_hip': (70, 110)
                        },
                        duration_range=(10, 30),     # Frames for descent
                        required_sequence=[]
                    ),
                    MovementPhase(
                        name='hold',
                        joint_angles={
                            'left_knee': (85, 95),   # Deep squat position
                            'right_knee': (85, 95),
                            'left_hip': (65, 75),
                            'right_hip': (65, 75)
                        },
                        duration_range=(5, 15),      # Frames for hold
                        required_sequence=['descent']
                    ),
                    MovementPhase(
                        name='ascent',
                        joint_angles={
                            'left_knee': (100, 175),  # Return to standing
                            'right_knee': (100, 175),
                            'left_hip': (100, 170),
                            'right_hip': (100, 170)
                        },
                        duration_range=(10, 30),
                        required_sequence=['hold']
                    )
                ]
            ),
            'lateral_raise': MovementPattern(
                name='lateral_raise',
                phases=[
                    MovementPhase(
                        name='raise',
                        joint_angles={
                            'left_shoulder': (30, 90),
                            'right_shoulder': (30, 90),
                            'left_elbow': (165, 180),  # Keep arm straight
                            'right_elbow': (165, 180)
                        },
                        duration_range=(15, 45),
                        required_sequence=[]
                    ),
                    MovementPhase(
                        name='lower',
                        joint_angles={
                            'left_shoulder': (10, 30),
                            'right_shoulder': (10, 30),
                            'left_elbow': (165, 180),
                            'right_elbow': (165, 180)
                        },
                        duration_range=(15, 45),
                        required_sequence=['raise']
                    )
                ]
            )
        }
        
        self.current_phase = None
        self.phase_frame_count = 0
        self.movement_history = []
        
    def analyze_frame(self, frame_data: Dict) -> Dict:
        """
        Analyze a single frame of movement data
        Returns detected movement patterns and their confidence scores
        """
        angles = frame_data['angles']
        velocities = frame_data['velocities']
        
        # Initialize results
        results = {
            'detected_movements': [],
            'current_phase': None,
            'form_issues': [],
            'metrics': {}
        }
        
        # Update movement history
        self.movement_history.append(frame_data)
        if len(self.movement_history) > 90:  # Keep 3 seconds at 30fps
            self.movement_history.pop(0)
            
        # Analyze each movement pattern
        for pattern_name, pattern in self.movement_patterns.items():
            pattern_results = self._analyze_pattern(pattern)
            if pattern_results['confidence'] > 0.6:  # Confidence threshold
                results['detected_movements'].append({
                    'name': pattern_name,
                    'phase': pattern_results['current_phase'],
                    'confidence': pattern_results['confidence'],
                    'form_score': pattern_results['form_score'],
                    'symmetry_score': pattern_results['symmetry_score']
                })
                
                # Add form issues if detected
                results['form_issues'].extend(pattern_results['form_issues'])
        
        # Calculate movement metrics
        results['metrics'] = self._calculate_metrics(frame_data)
        
        return results
    
    def _analyze_pattern(self, pattern: MovementPattern) -> Dict:
        """Analyze movement against a specific pattern"""
        if len(self.movement_history) < 2:
            return {
                'confidence': 0,
                'current_phase': None,
                'form_score': 0,
                'symmetry_score': 0,
                'form_issues': []
            }
            
        # Get most recent frames
        current_frame = self.movement_history[-1]
        prev_frame = self.movement_history[-2]
        
        # Initialize results
        form_issues = []
        
        # Check each phase
        for phase in pattern.phases:
            phase_match = self._check_phase_match(phase, current_frame['angles'])
            if phase_match['matches']:
                # Calculate form score
                form_score = self._calculate_form_score(
                    current_frame['angles'],
                    phase.joint_angles
                )
                
                # Calculate symmetry score if bilateral
                symmetry_score = self._calculate_symmetry_score(
                    current_frame['angles']
                ) if pattern.bilateral else 1.0
                
                # Check for form issues
                if form_score < 0.8:
                    form_issues.append(f"Poor form detected in {phase.name} phase")
                if pattern.bilateral and symmetry_score < 0.8:
                    form_issues.append("Significant asymmetry detected")
                
                return {
                    'confidence': phase_match['confidence'],
                    'current_phase': phase.name,
                    'form_score': form_score,
                    'symmetry_score': symmetry_score,
                    'form_issues': form_issues
                }
        
        return {
            'confidence': 0,
            'current_phase': None,
            'form_score': 0,
            'symmetry_score': 0,
            'form_issues': []
        }
        
    def _check_phase_match(self, phase: MovementPhase, angles: Dict) -> Dict:
        """Check if current angles match a movement phase"""
        matches = True
        total_confidence = 0
        num_angles = 0
        
        for joint, (min_angle, max_angle) in phase.joint_angles.items():
            if joint not in angles:
                continue
                
            # Extract just the angle value from the tuple
            current_angle = angles[joint][1]  # Get the second element (angle value)
            
            if not (min_angle <= current_angle <= max_angle):
                matches = False
                break
                
            # Calculate confidence based on how centered the angle is in the range
            range_center = (min_angle + max_angle) / 2
            range_size = max_angle - min_angle
            distance_from_center = abs(current_angle - range_center)
            confidence = 1 - (distance_from_center / (range_size / 2))
            
            total_confidence += confidence
            num_angles += 1
        
        avg_confidence = total_confidence / num_angles if num_angles > 0 else 0
        
        return {
            'matches': matches,
            'confidence': avg_confidence if matches else 0
        }

    def _calculate_form_score(self, current_angles: Dict, target_angles: Dict) -> float:
        """Calculate form score based on how well angles match the target"""
        total_score = 0
        num_angles = 0
        
        for joint, (min_angle, max_angle) in target_angles.items():
            if joint not in current_angles:
                continue
                
            current_angle = current_angles[joint][1]  # Get the second element (angle value)
            target_center = (min_angle + max_angle) / 2
            max_deviation = (max_angle - min_angle) / 2
            
            deviation = abs(current_angle - target_center)
            score = max(0, 1 - (deviation / max_deviation))
            
            total_score += score
            num_angles += 1
        
        return total_score / num_angles if num_angles > 0 else 0

    def _calculate_symmetry_score(self, angles: Dict) -> float:
        """Calculate bilateral symmetry score"""
        pairs = [
            ('left_knee', 'right_knee'),
            ('left_hip', 'right_hip'),
            ('left_shoulder', 'right_shoulder'),
            ('left_elbow', 'right_elbow')
        ]
        
        total_score = 0
        num_pairs = 0
        
        for left, right in pairs:
            if left in angles and right in angles:
                # Extract angle values from tuples
                left_angle = angles[left][1]
                right_angle = angles[right][1]
                diff = abs(left_angle - right_angle)
                score = max(0, 1 - (diff / 30))  # 30 degrees max difference
                total_score += score
                num_pairs += 1
        
        return total_score / num_pairs if num_pairs > 0 else 0
    
    def _calculate_metrics(self, frame_data: Dict) -> Dict:
        """Calculate additional movement metrics"""
        return {
            'movement_speed': np.mean(list(frame_data['velocities'].values())),
            'range_of_motion': {
                joint: max(angle) - min(angle)
                for joint, angle in frame_data['angles'].items()
            },
            'stability': self._calculate_stability(frame_data),
        }
    
    def _calculate_stability(self, frame_data: Dict) -> float:
        """Calculate movement stability score"""
        if len(self.movement_history) < 5:
            return 1.0
            
        # Calculate velocity variance over recent frames
        recent_velocities = [
            frame['velocities']
            for frame in self.movement_history[-5:]
        ]
        
        # Lower variance = more stable movement
        variance = np.var([
            np.mean(list(v.values()))
            for v in recent_velocities
        ])
        
        return max(0, 1 - (variance / 10))  # Scale variance to 0-1 score