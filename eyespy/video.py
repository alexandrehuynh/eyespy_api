import cv2
import numpy as np
from pathlib import Path
import asyncio
from typing import List, Tuple, Optional
from .config import settings

class VideoProcessor:
    def __init__(self):
        self.temp_dir = Path(settings.TEMP_DIR)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def extract_frames(
        self, 
        video_path: Path,
        target_fps: int = 30,
        max_duration: int = 10  # Max duration in seconds
    ) -> Tuple[List[np.ndarray], dict]:
        """
        Extract frames with intelligent frame sampling
        Returns: (frames, metadata)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame sampling
        target_frames = min(target_fps * max_duration, total_frames)
        sample_interval = max(1, total_frames // target_frames)
        
        frames = []
        frame_indices = []
        frame_count = 0
        
        try:
            while cap.isOpened() and len(frames) < target_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_interval == 0:
                    frames.append(frame)
                    frame_indices.append(frame_count)
                
                frame_count += 1
                
                # Allow other async operations
                if frame_count % 10 == 0:  # Process in small batches
                    await asyncio.sleep(0)
            
            metadata = {
                "original_fps": original_fps,
                "processed_fps": target_fps,
                "total_frames": total_frames,
                "sampled_frames": len(frames),
                "duration": duration,
                "frame_indices": frame_indices,
                "dimensions": (width, height)
            }
            
            return frames, metadata
            
        finally:
            cap.release()

# Update app/pose/mediapipe_estimator.py
import mediapipe as mp
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from ..models import Keypoint
import asyncio

class MediaPipeEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # Enable video mode
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    async def process_frames(
        self, 
        frames: List[np.ndarray],
        batch_size: int = 5
    ) -> List[Optional[List[Keypoint]]]:
        """Process multiple frames with batching"""
        all_keypoints = []
        
        # Process frames in batches
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_keypoints = []
            
            for frame in batch:
                keypoints = self._process_single_frame(frame)
                batch_keypoints.append(keypoints)
            
            all_keypoints.extend(batch_keypoints)
            
            # Yield control after each batch
            await asyncio.sleep(0)
        
        return all_keypoints
    
    def _process_single_frame(self, frame: np.ndarray) -> Optional[List[Keypoint]]:
        """Process a single frame"""
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

# Update app/main.py endpoint
@app.post("/api/v1/pose", response_model=PoseEstimationResponse)
async def process_video(
    video: UploadFile = File(...),
    target_fps: int = 30,
    max_duration: int = 10
) -> PoseEstimationResponse:
    """
    Process video for pose estimation with multi-frame support
    - target_fps: Target frames per second to process
    - max_duration: Maximum duration in seconds to process
    """
    # Validate video format
    if not any(video.filename.lower().endswith(fmt) 
              for fmt in settings.SUPPORTED_FORMATS):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Supported formats: {settings.SUPPORTED_FORMATS}"
        )
    
    video_processor = VideoProcessor()
    pose_estimator = MediaPipeEstimator()
    
    try:
        # Save uploaded file
        video_path = await video_processor.save_upload(video)
        if not video_path:
            raise HTTPException(
                status_code=500,
                detail="Failed to save video file"
            )
        
        # Extract frames
        frames, video_metadata = await video_processor.extract_frames(
            video_path,
            target_fps=target_fps,
            max_duration=max_duration
        )
        
        if not frames:
            raise HTTPException(
                status_code=400,
                detail="No frames could be extracted from video"
            )
        
        # Process all frames
        all_keypoints = await pose_estimator.process_frames(frames)
        
        # Cleanup
        await video_processor.cleanup(video_path)
        
        # Filter out None values and get the most recent valid keypoints
        valid_keypoints = [kp for kp in all_keypoints if kp is not None]
        latest_keypoints = valid_keypoints[-1] if valid_keypoints else None
        
        if not valid_keypoints:
            return PoseEstimationResponse(
                status=ProcessingStatus.COMPLETED,
                metadata={
                    **video_metadata,
                    "filename": video.filename,
                    "message": "No pose detected in any frame"
                }
            )
        
        return PoseEstimationResponse(
            status=ProcessingStatus.COMPLETED,
            keypoints=latest_keypoints,
            metadata={
                **video_metadata,
                "filename": video.filename,
                "frames_with_pose": len(valid_keypoints),
                "detection_rate": len(valid_keypoints) / len(frames)
            }
        )
        
    except Exception as e:
        if video_path:
            await video_processor.cleanup(video_path)
        return PoseEstimationResponse(
            status=ProcessingStatus.FAILED,
            error=str(e)
        )