import asyncio
import functools
from typing import Any, Callable, TypeVar, Coroutine
import logging
from .executor_service import get_executor

logger = logging.getLogger(__name__)

T = TypeVar('T')

async def run_in_executor(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run a blocking function in the shared thread pool executor.
    
    Args:
        func: The blocking function to run
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function execution
    """
    loop = asyncio.get_event_loop()
    
    # Use partial to combine the function with its arguments
    if kwargs:
        func_to_run = functools.partial(func, *args, **kwargs)
    else:
        func_to_run = lambda: func(*args)
    
    try:
        return await loop.run_in_executor(
            None, functools.partial(func_to_run)
        )
    except Exception as e:
        logger.error(f"Error executing {func.__name__} in executor: {e}")
        raise

async def gather_with_concurrency(n: int, *tasks) -> list:
    """
    Run tasks with a limit on concurrency.
    
    Args:
        n: Maximum number of concurrent tasks
        *tasks: Tasks to run
        
    Returns:
        List of results from all tasks
    """
    semaphore = asyncio.Semaphore(n)
    
    async def sem_task(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(*(sem_task(task) for task in tasks))

def run_sync(func: Callable[..., Coroutine]) -> Callable[..., Any]:
    """
    Decorator to run an async function synchronously.
    Useful for running async functions in non-async contexts.
    
    Args:
        func: Async function to run
        
    Returns:
        Wrapped function that runs the async function synchronously
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop is available, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(func(*args, **kwargs))
    
    return wrapper

def none_safe(func):
    """Decorator to safely handle None inputs.
    
    Prevents 'NoneType' object is not iterable errors by checking inputs
    before processing.
    
    Args:
        func: The function to decorate.
        
    Returns:
        A wrapped function that checks for None inputs.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if any positional args are None
        if any(arg is None for arg in args):
            logger.warning(f"None value passed to {func.__name__}, returning None")
            return None
            
        # Check if any keyword args are None for known iterable parameters
        for param_name, value in kwargs.items():
            if value is None and param_name in ('keypoints', 'points', 'coordinates', 'data'):
                logger.warning(f"None value for {param_name} passed to {func.__name__}, returning None")
                return None
                
        return func(*args, **kwargs)
    return wrapper