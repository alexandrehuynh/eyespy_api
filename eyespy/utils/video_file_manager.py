import os
import json
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from fastapi import UploadFile
from uuid import uuid4

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoFileManager:
    """
    Manages video file paths, storage, and metadata
    
    This component handles all file operations for video processing:
    - Generating unique paths for input/output files
    - Saving uploaded files
    - Saving metadata
    - Cleaning up temporary files
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the file manager with base directory
        
        Args:
            base_dir: Base directory for file storage (defaults to parent of current file)
        """
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        
        # Define folder structure
        self.upload_dir = self.base_dir / "uploads"
        self.processed_dir = self.base_dir / "processed"
        self.rendered_dir = self.processed_dir / "rendered"
        self.metadata_dir = self.processed_dir / "metadata"
        
        # Create folders if they don't exist
        for folder in [self.upload_dir, self.processed_dir, self.rendered_dir, self.metadata_dir]:
            folder.mkdir(parents=True, exist_ok=True)
    
    def generate_paths(self, file: UploadFile) -> Dict[str, Path]:
        """
        Generate paths for input and output files
        
        Args:
            file: Uploaded file
            
        Returns:
            Dictionary of paths for input, output, and metadata files
        """
        # Generate timestamp and unique ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid4())[:8]
        
        # Get file extension
        filename = file.filename or f"video_{timestamp}.mp4"
        file_stem = Path(filename).stem
        file_extension = Path(filename).suffix
        
        # Generate paths
        paths = {
            "input": self.upload_dir / f"{file_stem}_{timestamp}_{unique_id}{file_extension}",
            "metadata": self.metadata_dir / f"{file_stem}_{timestamp}_{unique_id}.json",
        }
        
        # Generate rendering output paths for different modes
        for mode in ["standard", "wireframe", "analysis", "xray"]:
            paths[mode] = self.rendered_dir / f"{mode}_{file_stem}_{timestamp}_{unique_id}{file_extension}"
        
        return paths
    
    async def save_upload(self, file: UploadFile) -> Path:
        """
        Save uploaded file to disk
        
        Args:
            file: Uploaded file
            
        Returns:
            Path where the file was saved
        """
        # Generate unique path
        input_path = self.generate_paths(file)["input"]
        
        # Read file content
        file_content = await file.read()
        
        # Save the file
        with open(input_path, "wb") as f:
            f.write(file_content)
        
        logger.info(f"Saved uploaded file to {input_path}")
        
        return input_path
    
    def save_metadata(self, metadata: Dict[str, Any], path: Path) -> None:
        """
        Save metadata to JSON file
        
        Args:
            metadata: Metadata to save
            path: Path to save metadata to
        """
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, "w") as f:
                json.dump(metadata, f, indent=4)
                
            logger.info(f"Saved metadata to {path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    async def cleanup(self, paths: List[Path]) -> None:
        """
        Clean up temporary files
        
        Args:
            paths: List of paths to clean up
        """
        for path in paths:
            try:
                if path.exists():
                    path.unlink()
                    logger.debug(f"Removed temporary file: {path}")
            except Exception as e:
                logger.error(f"Error removing file {path}: {e}")
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """
        Get information about a video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        import cv2
        
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            
            # Get properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Close video
            cap.release()
            
            return {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration": duration,
                "size_mb": video_path.stat().st_size / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {
                "error": str(e)
            }
    
    def generate_screenshots(
        self,
        video_path: Path,
        output_dir: Optional[Path] = None,
        count: int = 3
    ) -> List[Path]:
        """
        Generate screenshots from a video
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save screenshots (defaults to screenshots folder next to video)
            count: Number of screenshots to generate
            
        Returns:
            List of paths to generated screenshots
        """
        import cv2
        
        # Default output directory
        if output_dir is None:
            output_dir = video_path.parent / "screenshots"
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            
            # Get properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame indices
            if count == 1:
                indices = [frame_count // 2]  # Middle frame
            else:
                indices = [
                    int(i * frame_count / (count + 1))
                    for i in range(1, count + 1)
                ]
            
            # Generate screenshots
            screenshots = []
            for i, idx in enumerate(indices):
                # Set position
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Generate path
                screenshot_path = output_dir / f"screenshot_{i+1}.jpg"
                
                # Save frame
                cv2.imwrite(str(screenshot_path), frame)
                
                screenshots.append(screenshot_path)
            
            # Close video
            cap.release()
            
            return screenshots
        except Exception as e:
            logger.error(f"Error generating screenshots: {e}")
            return []