import time
import psutil
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    batch_times: List[float]
    memory_usage: List[float]
    fps_values: List[float]
    total_frames: int
    total_time: float
    
    @property
    def average_fps(self) -> float:
        return self.total_frames / self.total_time
    
    @property
    def average_batch_time(self) -> float:
        return np.mean(self.batch_times)
    
    @property
    def average_memory(self) -> float:
        return np.mean(self.memory_usage)
    
    @property
    def peak_memory(self) -> float:
        return max(self.memory_usage)
    
    def to_dict(self) -> Dict:
        return {
            "average_fps": self.average_fps,
            "average_batch_time": self.average_batch_time,
            "average_memory_percent": self.average_memory,
            "peak_memory_percent": self.peak_memory,
            "total_frames": self.total_frames,
            "total_time_seconds": self.total_time
        }

@dataclass
class PerformanceConfig:
    max_memory_percent: float = 80.0
    target_fps: float = 30.0
    batch_size_min: int = 10
    batch_size_max: int = 100
    cpu_threshold: float = 90.0
    adaptive_batch_size: bool = True

class PerformanceMonitor:
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.process = psutil.Process()
        self.start_time = time.time()
        self.frames_processed = 0
        self.current_batch_size = self.config.batch_size_min
        self._last_adjustment = time.time()
        
    async def check_resources(self) -> Dict:
        """Check system resources and adjust processing parameters"""
        cpu_percent = psutil.cpu_percent()
        memory_percent = self.process.memory_percent()
        
        metrics = {
            'cpu_usage': cpu_percent,
            'memory_usage': memory_percent,
            'batch_size': self.current_batch_size,
            'fps': self.get_current_fps()
        }
        
        # Log if resources are strained
        if cpu_percent > self.config.cpu_threshold:
            logger.warning(f"High CPU usage: {cpu_percent}%")
        if memory_percent > self.config.max_memory_percent:
            logger.warning(f"High memory usage: {memory_percent}%")
            
        # Adjust batch size if needed
        if self.config.adaptive_batch_size:
            await self._adjust_batch_size(cpu_percent, memory_percent)
            
        return metrics
    
    async def _adjust_batch_size(self, cpu_percent: float, memory_percent: float):
        """Dynamically adjust batch size based on system load"""
        now = time.time()
        # Only adjust every 5 seconds
        if now - self._last_adjustment < 5:
            return
            
        self._last_adjustment = now
        
        # Decrease batch size if resources are strained
        if cpu_percent > self.config.cpu_threshold or memory_percent > self.config.max_memory_percent:
            self.current_batch_size = max(
                self.config.batch_size_min,
                int(self.current_batch_size * 0.8)
            )
            logger.info(f"Decreased batch size to {self.current_batch_size}")
            
        # Increase batch size if resources are available
        elif cpu_percent < self.config.cpu_threshold * 0.7 and memory_percent < self.config.max_memory_percent * 0.7:
            self.current_batch_size = min(
                self.config.batch_size_max,
                int(self.current_batch_size * 1.2)
            )
            logger.info(f"Increased batch size to {self.current_batch_size}")
    
    def get_current_fps(self) -> float:
        """Calculate current FPS"""
        elapsed_time = time.time() - self.start_time
        return self.frames_processed / elapsed_time if elapsed_time > 0 else 0
    
    def frame_processed(self):
        """Call this when a frame is processed"""
        self.frames_processed += 1 