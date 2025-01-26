import cv2
import numpy as np
from pathlib import Path
import shutil
from typing import Optional
import asyncio
from .config import settings

class VideoProcessor:
    def __init__(self):
        self.temp_dir = Path(settings.TEMP_DIR)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_upload(self, file) -> Optional[Path]:
        """Save uploaded file to temporary directory"""
        temp_path = self.temp_dir / file.filename
        
        try:
            # Save the uploaded file
            with temp_path.open("wb") as buffer:
                content = await file.read()
                buffer.write(content)
            return temp_path
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e

    async def cleanup(self, file_path: Path):
        """Remove temporary file"""
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass