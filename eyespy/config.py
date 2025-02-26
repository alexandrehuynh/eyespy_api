from pydantic_settings import BaseSettings
from typing import Dict, Optional
import os

class Settings(BaseSettings):
    API_NAME: str = "EyeSpy API"
    VERSION: str = "0.1.0"
    
    # Video Processing Settings
    MAX_VIDEO_SIZE_MB: int = 100
    SUPPORTED_FORMATS: list[str] = ["mp4", "avi", "mov"]
    TEMP_DIR: str = "/tmp/eyespy"
    MAX_FRAMES_TO_PROCESS: Optional[int] = None  # Process the entire video
    TARGET_FPS: int = 15  # Reduced from 30 for better performance
    MAX_DIMENSION: int = 720  # Target 720p resolution
    
    # Frame Processing
    FRAME_INTERVAL: int = 2  # Process every nth frame (2 = skip every other frame)
    MINIMAL_PROCESSING: bool = False  # Set to True for absolute fastest processing
    
    # Confidence Thresholds
    GLOBAL_CONFIDENCE_THRESHOLD: float = 0.5
    # Different thresholds for different body parts
    KEYPOINT_THRESHOLDS: Dict[str, float] = {
        "NOSE": 0.5,
        "LEFT_EYE": 0.5,
        "RIGHT_EYE": 0.5,
        "LEFT_SHOULDER": 0.6,
        "RIGHT_SHOULDER": 0.6,
        "LEFT_ELBOW": 0.5,
        "RIGHT_ELBOW": 0.5,
        "LEFT_WRIST": 0.4,
        "RIGHT_WRIST": 0.4,
        "LEFT_HIP": 0.6,
        "RIGHT_HIP": 0.6,
        "LEFT_KNEE": 0.5,
        "RIGHT_KNEE": 0.5,
        "LEFT_ANKLE": 0.4,
        "RIGHT_ANKLE": 0.4
    }

    # MediaPipe Settings
    MEDIAPIPE_MODEL_COMPLEXITY: int = 1  # 0=Lite, 1=Full, 2=Heavy
    MEDIAPIPE_SMOOTH_LANDMARKS: bool = True
    
    # Performance Settings
    BATCH_SIZE: int = 30  # Reduced from 50 for more consistent processing
    MIN_MEMORY_THRESHOLD: float = 20.0  # Minimum free memory percentage
    MAX_MEMORY_THRESHOLD: float = 80.0  # Maximum memory usage percentage
    THREAD_POOL_WORKERS: Optional[int] = None  # None = auto-detect based on CPU
    ENABLE_PERFORMANCE_LOGGING: bool = True
    
    # Quality Assessment
    ENABLE_QUALITY_ASSESSMENT: bool = True
    QUALITY_THRESHOLD: float = 0.4  # Minimum quality score to process a frame
    
    # Monitoring Settings
    LOG_BATCH_METRICS: bool = True
    PERFORMANCE_LOG_INTERVAL: int = 100  # Log every N frames
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR

    # Processing Modes
    PROCESSING_MODES: Dict[str, Dict[str, any]] = {
        "fast": {
            "FRAME_INTERVAL": 3,
            "MEDIAPIPE_MODEL_COMPLEXITY": 0,
            "QUALITY_THRESHOLD": 0.3,
            "BATCH_SIZE": 20,
            "TARGET_FPS": 10
        },
        "balanced": {
            "FRAME_INTERVAL": 2,
            "MEDIAPIPE_MODEL_COMPLEXITY": 1,
            "QUALITY_THRESHOLD": 0.4,
            "BATCH_SIZE": 30,
            "TARGET_FPS": 15
        },
        "quality": {
            "FRAME_INTERVAL": 1,
            "MEDIAPIPE_MODEL_COMPLEXITY": 2,
            "QUALITY_THRESHOLD": 0.5,
            "BATCH_SIZE": 30,
            "TARGET_FPS": 30
        }
    }

    def get_mode_settings(self, mode: str = "balanced") -> Dict[str, any]:
        """Get settings for a specific processing mode"""
        if mode not in self.PROCESSING_MODES:
            mode = "balanced"
        return self.PROCESSING_MODES[mode]

    class Config:
        env_file = ".env"
        env_prefix = "EYESPY_"

settings = Settings()

# Update logging level based on settings
import logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)