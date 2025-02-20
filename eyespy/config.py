from pydantic_settings import BaseSettings
from typing import Dict, Optional

class Settings(BaseSettings):
    API_NAME: str = "EyeSpy API"
    VERSION: str = "0.1.0"
    
    # Video Processing Settings
    MAX_VIDEO_SIZE_MB: int = 100
    SUPPORTED_FORMATS: list[str] = ["mp4", "avi", "mov"]
    TEMP_DIR: str = "/tmp/eyespy"
    MAX_FRAMES_TO_PROCESS: Optional[int] = None  # Process the entire video
    TARGET_FPS: int = 30
    MAX_DIMENSION: int = 720  # Target 720p resolution
    
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

    # Performance Settings
    BATCH_SIZE: int = 50
    MIN_MEMORY_THRESHOLD: float = 20.0  # Minimum free memory percentage
    MAX_MEMORY_THRESHOLD: float = 80.0  # Maximum memory usage percentage
    THREAD_POOL_WORKERS: Optional[int] = None  # None = auto-detect based on CPU
    ENABLE_PERFORMANCE_LOGGING: bool = True
    
    # Monitoring Settings
    LOG_BATCH_METRICS: bool = True
    PERFORMANCE_LOG_INTERVAL: int = 100  # Log every N frames

    class Config:
        env_file = ".env"

settings = Settings()