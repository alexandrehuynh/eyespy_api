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

class PerformanceMonitor:
    def __init__(self):
        self.batch_times: List[float] = []
        self.memory_usage: List[float] = []
        self.fps_values: List[float] = []
        self.start_time = time.time()
        self.total_frames = 0
        
    def record_batch(self, batch_size: int, batch_time: float):
        self.batch_times.append(batch_time)
        self.memory_usage.append(psutil.Process().memory_percent())
        self.fps_values.append(batch_size / batch_time)
        self.total_frames += batch_size
        
    def get_metrics(self) -> PerformanceMetrics:
        return PerformanceMetrics(
            batch_times=self.batch_times,
            memory_usage=self.memory_usage,
            fps_values=self.fps_values,
            total_frames=self.total_frames,
            total_time=time.time() - self.start_time
        )
    
    def log_batch_metrics(self, batch_index: int):
        logger.info(
            f"Batch {batch_index}: "
            f"FPS: {self.fps_values[-1]:.1f}, "
            f"Memory: {self.memory_usage[-1]:.1f}%, "
            f"Time: {self.batch_times[-1]:.3f}s"
        ) 