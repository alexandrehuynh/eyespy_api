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
    loop = asyncio.get_running_loop()
    
    # Use partial to combine the function with its arguments
    if kwargs:
        func_to_run = functools.partial(func, *args, **kwargs)
    else:
        func_to_run = lambda: func(*args)
    
    try:
        return await loop.run_in_executor(get_executor(), func_to_run)
    except Exception as e:
        logger.error(f"Error executing {func.__name__} in executor: {str(e)}")
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