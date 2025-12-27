from functools import wraps
from .logger import get_logger, log_exception

def log_errors(context: str | None = None, reraise: bool = True):
    """
    Decorator to log all exceptions inside a function
    args:
        context: Optional context string for logs
        reraise: Whether to re-raise the exception after logging
    """
    
    def decorator(func):
        logger = get_logger()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            try: 
                return func(*args, **kwargs)
            except Exception as e:
                log_exception(logger, e, context=context or func.__name__)
                if reraise:
                    raise 
                return None
        return wrapper
        
    return decorator


# To use in any specific file or function 

# from src.logging.error_handler import log_errors

# inside a function:
# @log_errors(context="specific_function")
# for eg: @log_errors(context="BayesianOptimizer.observe")    
#         @log_errors(context="sandbox_test.main")
        
# currently going with global_exception
