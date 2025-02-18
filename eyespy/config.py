from pydantic_settings import BaseSettings
from typing import Dict

class Settings(BaseSettings):
    API_NAME: str = "EyeSpy API"
    VERSION: str = "0.1.0"
    
    # Video Processing Settings
    MAX_VIDEO_SIZE_MB: int = 100
    MAX_FRAMES_TO_PROCESS: int = 150  # About 5 seconds at 30fps
    SUPPORTED_FORMATS: list[str] = ["mp4", "avi", "mov"]
    TEMP_DIR: str = "/tmp/eyespy"
    TARGET_RESOLUTION: tuple = (1280, 720)  # 720p
    
    # Processing Timeouts
    FRAME_PROCESSING_TIMEOUT: int = 180  # 3 minutes max
    
    # Batch Processing
    BATCH_SIZE: int = 10
    
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

    class Config:
        env_file = ".env"

settings = Settings()