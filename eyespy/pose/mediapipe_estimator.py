import mediapipe as mp
import numpy as np
import cv2
from typing import List, Optional
from ..models import Keypoint
from ..config import settings

class MediaPipeEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame: np.ndarray) -> Optional[List[Keypoint]]:
        """Process a single frame and return keypoints"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            if not results.pose_landmarks:
                return None
            
            # Convert landmarks to our Keypoint model
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
            
            return keypoints
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return None

    def __del__(self):
        """Cleanup MediaPipe resources"""
        self.pose.close()
