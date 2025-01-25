from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_NAME: str = "EyeSpy API"
    VERSION: str = "0.1.0"
    
    # Video Processing Settings
    MAX_VIDEO_SIZE_MB: int = 100
    SUPPORTED_FORMATS: list[str] = ["mp4", "avi", "mov"]
    TEMP_DIR: str = "/tmp/eyespy"

    class Config:
        env_file = ".env"

settings = Settings()