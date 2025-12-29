import sys
from .logger import get_logger, log_exception

def install_global_exception_handler():
    logger = get_logger()
    
    def handle_excpetion(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        log_exception(logger, exc_value, context="UNCAUGHT EXCEPTION")
        
    sys.excepthook = handle_excpetion
    
    
# this will help to call once at program start:
    
# from src.logging.global_exception import install_global_exception_handler
# install_global_exception_handler()