import logging
import os
import sys
import traceback
from datetime import datetime


LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)


def get_logger(name: str = "AutoTuneNet") -> logging.Logger:
    log = logging.getLogger(name)

    if log.handlers:
        return log

    log.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s | %(levelname)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setFormatter(formatter)

    log.addHandler(stream_handler)
    log.addHandler(file_handler)

    return log

def log_exception(logger: logging.Logger, exc: Exception, context: str | None = None) -> None:
    error_type = type(exc).__name__
    error_msg = str(exc)
    tb = traceback.format_exc()
    
    header = f"EXCEPTION OCCURED [{error_type}]"
    if context:
        header += f" | Context: {context}"
        
    logger.error(header)
    logger.error(f"Message: {error_msg}")
    logger.error("Traceback:")
    logger.error(tb)