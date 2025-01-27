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

    async def extract_frames(self, video_path: Path, max_frames: int = 30):
        """Extract frames from video file"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Calculate frame interval to get ~max_frames frames
        interval = max(1, total_frames // max_frames)
        
        frames = []
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % interval == 0:
                    frames.append(frame)
                    
                frame_count += 1
                
                # Allow other async operations
                await asyncio.sleep(0)
                
            return frames, fps
            
        finally:
            cap.release()