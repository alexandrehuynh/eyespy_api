import mediapipe as mp
import numpy as np
from typing import List, Optional, Dict
from ..models import Keypoint
from ..config import settings
import asyncio
import cv2
from .validation import PoseValidator
from .confidence import ConfidenceAssessor
from .tracker import ConfidenceTracker


class MediaPipeEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=settings.GLOBAL_CONFIDENCE_THRESHOLD,
            min_tracking_confidence=settings.GLOBAL_CONFIDENCE_THRESHOLD
        )
        self.keypoint_thresholds = settings.KEYPOINT_THRESHOLDS
        self.validator = PoseValidator()
        self.confidence_tracker = ConfidenceTracker()


    def _apply_threshold(self, keypoint: Keypoint) -> Optional[Keypoint]:
        """Apply confidence thresholding to a single keypoint"""
        # Get threshold for this keypoint type
        threshold = self.keypoint_thresholds.get(
            keypoint.name, 
            settings.GLOBAL_CONFIDENCE_THRESHOLD
        )
        
        # Return keypoint only if confidence exceeds threshold
        if keypoint.confidence >= threshold:
            return keypoint
        return None

    def _filter_keypoints(self, keypoints: List[Keypoint]) -> List[Keypoint]:
        """Filter keypoints based on enhanced confidence assessment"""
        if not keypoints:
            return []
            
        # Prepare confidence assessor
        confidence_assessor = ConfidenceAssessor()
        
        # Create dictionaries for positions and confidences
        positions = {kp.name: (kp.x, kp.y) for kp in keypoints}
        confidences = {kp.name: kp.confidence for kp in keypoints}
        
        # Filter keypoints
        filtered_keypoints = []
        for kp in keypoints:
            adjusted_confidence, is_valid = confidence_assessor.assess_keypoint(
                kp.name,
                kp.confidence,
                (kp.x, kp.y),
                positions,
                confidences
            )
            
            if is_valid:
                # Create new keypoint with adjusted confidence
                filtered_keypoints.append(
                    Keypoint(
                        x=kp.x,
                        y=kp.y,
                        confidence=adjusted_confidence,
                        name=kp.name
                    )
                )
        
        return filtered_keypoints

    def _validate_pose(self, keypoints: List[Keypoint]) -> bool:
        """Validate if enough keypoints are detected for a valid pose"""
        # Define critical keypoints for a valid pose
        critical_points = {
            "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "LEFT_HIP", "RIGHT_HIP"
        }
        
        # Check if we have enough critical points
        detected_critical = {
            kp.name for kp in keypoints 
            if kp.name in critical_points
        }
        
        return len(detected_critical) >= 3  # At least 3 critical points needed

    def _process_single_frame(self, frame: np.ndarray) -> Optional[List[Keypoint]]:
        """Process a single frame with confidence thresholding and pose validation"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            if not results.pose_landmarks:
                return None
            
            # Convert landmarks to keypoints
            keypoints = []
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                keypoints.append(
                    Keypoint(
                        x=landmark.x,
                        y=landmark.y,
                        confidence=landmark.visibility,
                        name=self.mp_pose.PoseLandmark(idx).name
                    )
                )
            
            # Apply confidence thresholding
            filtered_keypoints = self._filter_keypoints(keypoints)
            
            # Validate if we have enough keypoints for a valid pose
            if not self._validate_pose(filtered_keypoints):
                return None

            # Apply additional pose validation
            is_valid, validation_metrics = self.validator.validate_pose(filtered_keypoints)
            
            # Store validation metrics for use in response
            self.frame_metadata = validation_metrics
            
            # Return keypoints only if pose is valid
            if is_valid:
                return filtered_keypoints
                
            return None
                
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return None

    async def process_frames(
        self,
        frames: List[np.ndarray],
        batch_size: int = 5
    ) -> List[Optional[List[Keypoint]]]:
        """Process frames with confidence tracking"""
        all_keypoints = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_keypoints = []
            
            for frame in batch:
                # Get initial keypoints
                keypoints = self._process_single_frame(frame)
                
                if keypoints:
                    # Convert to format for tracker
                    kp_dict = {
                        kp.name: (kp.x, kp.y, kp.confidence)
                        for kp in keypoints
                    }
                    
                    # Update tracking
                    tracked_kp = await self.confidence_tracker.update(kp_dict)
                    
                    # Convert back to keypoints
                    keypoints = [
                        Keypoint(
                            name=name,
                            x=pos[0],
                            y=pos[1],
                            confidence=pos[2]
                        )
                        for name, pos in tracked_kp.items()
                    ]
                
                batch_keypoints.append(keypoints)
            
            all_keypoints.extend(batch_keypoints)
            await asyncio.sleep(0)
        
        return all_keypoints