import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from ..models import Keypoint
from ..utils.executor_service import get_executor
from ..utils.async_utils import run_in_executor, gather_with_concurrency, none_safe

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MovementPhase:
    """Represents a phase of a movement pattern with angle constraints"""
    name: str
    joint_angles: Dict[str, Tuple[float, float]]  # (min, max) angles for each joint
    duration_range: Tuple[int, int]  # (min, max) frames
    required_sequence: List[str]  # Required preceding phases

@dataclass
class MovementPattern:
    """Represents a complete movement pattern with multiple phases"""
    name: str
    phases: List[MovementPhase]
    bilateral: bool = True  # Whether movement should be symmetric


class MovementAnalyzer:
    """
    Analyzes movement patterns in pose keypoints
    
    This class identifies movement patterns like squats, lunges, etc.,
    and provides form assessments based on joint angles and motion.
    It uses the shared executor for parallel processing.
    """
    
    def __init__(self):
        """Initialize the movement analyzer with predefined movement patterns"""
        # Define movement patterns
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
            'lunge': MovementPattern(
                name='lunge',
                phases=[
                    MovementPhase(
                        name='descent',
                        joint_angles={
                            'left_knee': (80, 100),
                            'right_knee': (80, 100),
                            'left_hip': (80, 120),
                            'right_hip': (80, 120)
                        },
                        duration_range=(10, 30),
                        required_sequence=[]
                    ),
                    MovementPhase(
                        name='hold',
                        joint_angles={
                            'left_knee': (75, 90),
                            'right_knee': (75, 90),
                            'left_hip': (75, 100),
                            'right_hip': (75, 100)
                        },
                        duration_range=(5, 15),
                        required_sequence=['descent']
                    ),
                    MovementPhase(
                        name='ascent',
                        joint_angles={
                            'left_knee': (100, 170),
                            'right_knee': (100, 170),
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
                            'left_shoulder': (80, 100),
                            'right_shoulder': (80, 100),
                            'left_elbow': (160, 180),
                            'right_elbow': (160, 180)
                        },
                        duration_range=(10, 30),
                        required_sequence=[]
                    ),
                    MovementPhase(
                        name='hold',
                        joint_angles={
                            'left_shoulder': (85, 95),
                            'right_shoulder': (85, 95),
                            'left_elbow': (165, 180),
                            'right_elbow': (165, 180)
                        },
                        duration_range=(5, 15),
                        required_sequence=['raise']
                    ),
                    MovementPhase(
                        name='lower',
                        joint_angles={
                            'left_shoulder': (30, 70),
                            'right_shoulder': (30, 70),
                            'left_elbow': (160, 180),
                            'right_elbow': (160, 180)
                        },
                        duration_range=(10, 30),
                        required_sequence=['hold']
                    )
                ]
            )
        }
        
        # State tracking
        self.current_phase = None
        self.phase_frame_count = 0
        self.movement_history = []
        self.detected_reps = {pattern: 0 for pattern in self.movement_patterns}
    
    async def analyze_frame(self, angles: Dict[str, float], frame_index: int) -> Dict[str, Any]:
        """
        Analyze a single frame of pose data using async processing
        
        Args:
            angles: Dictionary of joint angles
            frame_index: Index of the current frame
            
        Returns:
            Analysis results for the frame
        """
        # Use the shared executor for CPU-bound processing
        return await run_in_executor(
            self._analyze_frame_sync,
            angles,
            frame_index
        )
    
    def _analyze_frame_sync(self, angles: Dict[str, float], frame_index: int) -> Dict[str, Any]:
        """
        Synchronous implementation of frame analysis
        
        Args:
            angles: Dictionary of joint angles
            frame_index: Index of the current frame
            
        Returns:
            Analysis results for the frame
        """
        # Update movement history
        self.movement_history.append({
            'angles': angles,
            'frame_index': frame_index,
            'timestamp': time.time()
        })
        
        # Limit history length
        if len(self.movement_history) > 90:  # Keep 3 seconds at 30fps
            self.movement_history.pop(0)
        
        # Initialize results
        results = {
            'detected_movements': [],
            'current_phase': None,
            'form_issues': [],
            'metrics': {}
        }
        
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
        results['metrics'] = self._calculate_metrics(angles)
        
        return results
    
    def _analyze_pattern(self, pattern: MovementPattern) -> Dict[str, Any]:
        """
        Analyze movement against a specific pattern
        
        Args:
            pattern: Movement pattern to analyze
            
        Returns:
            Analysis results for the pattern
        """
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
    
    def _check_phase_match(self, phase: MovementPhase, angles: Dict[str, float]) -> Dict[str, Any]:
        """
        Check if current angles match a movement phase
        
        Args:
            phase: Movement phase to check
            angles: Dictionary of joint angles
            
        Returns:
            Match results
        """
        matches = True
        total_confidence = 0
        num_angles = 0
        
        for joint, (min_angle, max_angle) in phase.joint_angles.items():
            if joint not in angles:
                continue
                
            current_angle = angles[joint]
            
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
    
    def _calculate_form_score(self, current_angles: Dict[str, float], target_angles: Dict[str, Tuple[float, float]]) -> float:
        """
        Calculate form score based on how well angles match the target
        
        Args:
            current_angles: Dictionary of current joint angles
            target_angles: Dictionary of target joint angle ranges
            
        Returns:
            Form score (0-1)
        """
        total_score = 0
        num_angles = 0
        
        for joint, (min_angle, max_angle) in target_angles.items():
            if joint not in current_angles:
                continue
                
            current_angle = current_angles[joint]
            target_center = (min_angle + max_angle) / 2
            max_deviation = (max_angle - min_angle) / 2
            
            deviation = abs(current_angle - target_center)
            score = max(0, 1 - (deviation / max_deviation))
            
            total_score += score
            num_angles += 1
        
        return total_score / num_angles if num_angles > 0 else 0
    
    def _calculate_symmetry_score(self, angles: Dict[str, float]) -> float:
        """
        Calculate bilateral symmetry score
        
        Args:
            angles: Dictionary of joint angles
            
        Returns:
            Symmetry score (0-1)
        """
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
                left_angle = angles[left]
                right_angle = angles[right]
                diff = abs(left_angle - right_angle)
                score = max(0, 1 - (diff / 30))  # 30 degrees max difference
                total_score += score
                num_pairs += 1
        
        return total_score / num_pairs if num_pairs > 0 else 0
    
    def _calculate_metrics(self, angles: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate additional movement metrics
        
        Args:
            angles: Dictionary of joint angles
            
        Returns:
            Dictionary of movement metrics
        """
        # Calculate range of motion for each joint
        rom = {}
        
        if len(self.movement_history) >= 2:
            for joint in angles.keys():
                angle_history = [
                    frame['angles'].get(joint)
                    for frame in self.movement_history
                    if joint in frame['angles']
                ]
                if angle_history:
                    rom[joint] = max(angle_history) - min(angle_history)
        
        return {
            'range_of_motion': rom,
            'stability': self._calculate_stability(angles),
        }
    
    def _calculate_stability(self, angles: Dict[str, float]) -> float:
        """
        Calculate movement stability score
        
        Args:
            angles: Dictionary of joint angles
            
        Returns:
            Stability score (0-1)
        """
        if len(self.movement_history) < 5:
            return 1.0
            
        # Calculate angle variance over recent frames
        recent_angles = [
            frame['angles']
            for frame in self.movement_history[-5:]
        ]
        
        # Get joints that are present in all frames
        common_joints = set(angles.keys())
        for frame_angles in recent_angles:
            common_joints &= set(frame_angles.keys())
        
        if not common_joints:
            return 1.0
        
        # Calculate variance for each joint
        variances = []
        for joint in common_joints:
            angle_values = [frame[joint] for frame in recent_angles if joint in frame]
            if len(angle_values) >= 3:  # Need at least 3 values for meaningful variance
                variances.append(np.var(angle_values))
        
        if not variances:
            return 1.0
        
        # Average variance (lower is more stable)
        avg_variance = np.mean(variances)
        
        # Convert to 0-1 score (higher is more stable)
        return max(0, 1 - (avg_variance / 500))
    
    async def analyze_sequence(self, keypoints_per_frame: List[List[Keypoint]]) -> Dict[str, Any]:
        """
        Analyze a sequence of frames for movement patterns
        
        Args:
            keypoints_per_frame: List of keypoints for each frame
            
        Returns:
            Comprehensive movement analysis
        """
        # Process frames in parallel with concurrency control
        angle_calculation_tasks = []
        
        for frame_idx, keypoints in enumerate(keypoints_per_frame):
            task = self._calculate_angles_from_keypoints(keypoints, frame_idx)
            angle_calculation_tasks.append(task)
        
        angles_per_frame = await gather_with_concurrency(
            4,  # Process up to 4 frames at a time
            *angle_calculation_tasks
        )
        
        # Process each frame's angles to detect movements
        frame_analysis_tasks = []
        
        for frame_idx, angles in enumerate(angles_per_frame):
            if angles:  # Skip frames with no valid angles
                task = self.analyze_frame(angles, frame_idx)
                frame_analysis_tasks.append(task)
        
        frame_analyses = await gather_with_concurrency(
            4,  # Process up to 4 frames at a time
            *frame_analysis_tasks
        )
        
        # Aggregate results
        return await run_in_executor(
            self._aggregate_analysis,
            frame_analyses
        )
    
    async def _calculate_angles_from_keypoints(self, keypoints: List[Keypoint], frame_idx: int) -> Dict[str, float]:
        """
        Calculate joint angles from keypoints
        
        Args:
            keypoints: List of keypoints for a frame
            frame_idx: Index of the current frame
            
        Returns:
            Dictionary of joint angles
        """
        return await run_in_executor(
            self._calculate_angles_from_keypoints_sync,
            keypoints,
            frame_idx
        )
    
    @none_safe
    def _calculate_angles_from_keypoints_sync(self, keypoints: List[Keypoint], frame_idx: int) -> Dict[str, float]:
        """
        Synchronous implementation of angle calculation from keypoints
        """
        # Define angle configurations
        angle_configs = {
            'left_elbow': ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
            'right_elbow': ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
            'left_shoulder': ('LEFT_ELBOW', 'LEFT_SHOULDER', 'LEFT_HIP'),
            'right_shoulder': ('RIGHT_ELBOW', 'RIGHT_SHOULDER', 'RIGHT_HIP'),
            'left_hip': ('LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'),
            'right_hip': ('RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'),
            'left_knee': ('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),
            'right_knee': ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE')
        }

        if keypoints is None or not keypoints:  # Check if None or empty
            return {}  # Return empty dictionary instead of list
        
        # Convert keypoints to dictionary for easier access
        keypoint_dict = {kp.name: kp for kp in keypoints}
        
        # Calculate angles
        angles = {}
        for angle_name, (p1_name, p2_name, p3_name) in angle_configs.items():
            if all(p_name in keypoint_dict for p_name in [p1_name, p2_name, p3_name]):
                p1 = keypoint_dict[p1_name]
                p2 = keypoint_dict[p2_name]
                p3 = keypoint_dict[p3_name]
                
                # Skip if confidence is too low
                if p1.confidence < 0.5 or p2.confidence < 0.5 or p3.confidence < 0.5:
                    continue
                
                # Calculate angle
                angle = self._calculate_angle(p1, p2, p3)
                angles[angle_name] = angle
        
        return angles
    
    def _calculate_angle(self, p1: Keypoint, p2: Keypoint, p3: Keypoint) -> float:
        """
        Calculate angle between three points
        
        Args:
            p1: First keypoint
            p2: Second keypoint (apex of angle)
            p3: Third keypoint
            
        Returns:
            Angle in degrees
        """
        # Convert to numpy arrays
        a = np.array([p1.x, p1.y])
        b = np.array([p2.x, p2.y])
        c = np.array([p3.x, p3.y])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate cosine of angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
        
        # Calculate angle
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _aggregate_analysis(self, frame_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate frame analyses into overall video analysis
        
        Args:
            frame_analyses: List of analysis results for each frame
            
        Returns:
            Aggregated analysis results
        """
        # Count detected movements
        movement_counts = {}
        for analysis in frame_analyses:
            for movement in analysis['detected_movements']:
                movement_name = movement['name']
                if movement_name not in movement_counts:
                    movement_counts[movement_name] = {
                        'count': 0,
                        'phases': {},
                        'form_scores': [],
                        'symmetry_scores': []
                    }
                
                movement_counts[movement_name]['count'] += 1
                
                phase = movement['phase']
                if phase:
                    movement_counts[movement_name]['phases'][phase] = movement_counts[movement_name]['phases'].get(phase, 0) + 1
                
                movement_counts[movement_name]['form_scores'].append(movement['form_score'])
                movement_counts[movement_name]['symmetry_scores'].append(movement['symmetry_score'])
        
        # Calculate repetitions
        repetitions = {}
        for movement_name, data in movement_counts.items():
            phases = data['phases']
            
            # A complete repetition requires all phases
            if 'ascent' in phases and 'descent' in phases:
                # A rep is counted as the minimum of ascent and descent phases
                reps = min(phases.get('ascent', 0), phases.get('descent', 0))
            else:
                reps = 0
            
            # Calculate average scores
            avg_form_score = np.mean(data['form_scores']) if data['form_scores'] else 0
            avg_symmetry_score = np.mean(data['symmetry_scores']) if data['symmetry_scores'] else 0
            
            repetitions[movement_name] = {
                'reps': reps,
                'avg_form_score': avg_form_score,
                'avg_symmetry_score': avg_symmetry_score
            }
        
        # Aggregate form issues
        form_issues = []
        for analysis in frame_analyses:
            for issue in analysis['form_issues']:
                if issue not in form_issues:
                    form_issues.append(issue)
        
        # Aggregate metrics
        rom_data = {}
        stability_scores = []
        
        for analysis in frame_analyses:
            metrics = analysis['metrics']
            
            # Combine range of motion data
            for joint, rom in metrics.get('range_of_motion', {}).items():
                if joint not in rom_data:
                    rom_data[joint] = []
                rom_data[joint].append(rom)
            
            # Collect stability scores
            if 'stability' in metrics:
                stability_scores.append(metrics['stability'])
        
        # Calculate average range of motion
        avg_rom = {
            joint: np.mean(values) for joint, values in rom_data.items()
        }
        
        # Calculate average stability
        avg_stability = np.mean(stability_scores) if stability_scores else 1.0
        
        return {
            'repetitions': repetitions,
            'form_issues': form_issues,
            'metrics': {
                'range_of_motion': avg_rom,
                'stability': avg_stability
            },
            'movement_quality': self._assess_movement_quality(repetitions, form_issues)
        }
    
    def _assess_movement_quality(self, repetitions: Dict[str, Dict[str, Any]], form_issues: List[str]) -> Dict[str, Any]:
        """
        Assess overall movement quality
        
        Args:
            repetitions: Dictionary of repetition data
            form_issues: List of form issues
            
        Returns:
            Movement quality assessment
        """
        # Calculate overall scores
        form_scores = []
        symmetry_scores = []
        
        for movement, data in repetitions.items():
            form_scores.append(data['avg_form_score'])
            symmetry_scores.append(data['avg_symmetry_score'])
        
        avg_form_score = np.mean(form_scores) if form_scores else 0
        avg_symmetry_score = np.mean(symmetry_scores) if symmetry_scores else 0
        
        # More form issues means lower quality
        issue_penalty = min(1.0, len(form_issues) * 0.1)
        
        # Calculate overall quality score
        quality_score = (avg_form_score * 0.6 + avg_symmetry_score * 0.4) * (1 - issue_penalty)
        
        # Generate quality assessment
        if quality_score >= 0.8:
            assessment = "Excellent"
        elif quality_score >= 0.6:
            assessment = "Good"
        elif quality_score >= 0.4:
            assessment = "Fair"
        else:
            assessment = "Poor"
        
        # Generate recommendations
        recommendations = []
        
        if avg_form_score < 0.6:
            recommendations.append("Focus on proper form and technique")
        if avg_symmetry_score < 0.6:
            recommendations.append("Work on bilateral balance and symmetry")
        if issue_penalty > 0.3:
            recommendations.append("Address specific form issues highlighted during the movement")
        
        return {
            'score': quality_score,
            'assessment': assessment,
            'recommendations': recommendations
        }