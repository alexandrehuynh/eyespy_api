import os
import psutil
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import atexit

# Set up logging
logger = logging.getLogger(__name__)

class ExecutorService:
    """
    Singleton class that provides a shared thread pool executor for the entire application.
    """
    _instance = None
    _executor = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ExecutorService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, max_workers: Optional[int] = None, name: str = "default"):
        if self._initialized:
            return
            
        # Calculate optimal number of workers if not specified
        if max_workers is None:
            # Use physical cores by default
            cpu_count = psutil.cpu_count(logical=False)
            # If can't determine physical cores, use logical cores / 2
            if cpu_count is None:
                cpu_count = max(1, psutil.cpu_count(logical=True) // 2)
                
            # Leave at least 1 core free for the main application
            max_workers = max(2, cpu_count - 1)
            
        logger.info(f"Initializing ExecutorService with {max_workers} workers")
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=name)
        self._initialized = True
        self._max_workers = max_workers
        self._name = name
        
        # Register shutdown handler
        atexit.register(self.shutdown)

    @property
    def executor(self) -> ThreadPoolExecutor:
        """Get the thread pool executor instance"""
        return self._executor
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor service"""
        if self._executor:
            logger.info(f"Shutting down ExecutorService ({self._name})")
            try:
                self._executor.shutdown(wait=wait)
                self._executor = None
                self._initialized = False
            except Exception as e:
                logger.error(f"Error shutting down ExecutorService: {str(e)}")

# Create a default shared executor service
default_executor = ExecutorService()

def get_executor() -> ThreadPoolExecutor:
    """Get the default shared executor instance"""
    return default_executor.executor