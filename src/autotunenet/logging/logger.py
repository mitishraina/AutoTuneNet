"""The AutoTuneNet logger: console output by default, optional log file."""
import logging
import os
import traceback
from datetime import datetime

LOGGER_NAME = "AutoTuneNet"
_FORMATTER = logging.Formatter("[%(asctime)s] %(name)s | %(levelname)s | %(message)s")


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """
    Return the AutoTuneNet logger.

    Importing AutoTuneNet has no side effects: nothing is written to disk
    unless file logging is enabled, either with enable_file_logging() or by
    setting the AUTOTUNENET_LOG_FILE environment variable to a file path.
    Messages go to the console once (the logger does not propagate to the
    root logger, so they are not printed twice).
    """
    log = logging.getLogger(name)
    if getattr(log, "_autotunenet_configured", False):
        return log

    log.setLevel(logging.INFO)
    log.propagate = False
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(_FORMATTER)
    log.addHandler(stream_handler)
    log._autotunenet_configured = True

    env_path = os.environ.get("AUTOTUNENET_LOG_FILE")
    if env_path:
        enable_file_logging(env_path, name=name)
    return log


def enable_file_logging(path: str | None = None, name: str = LOGGER_NAME) -> str:
    """
    Also write logs to a file and return its path.

    Calling it again with the same path does not add a second handler.

    Args:
        path: Log file path. Defaults to `./logs/<timestamp>.log`; the
            directory is created on demand.
        name: Logger name.

    Returns:
        The path written to.
    """
    if path is None:
        path = os.path.join("logs", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    log = get_logger(name)
    target = os.path.abspath(path)
    for handler in log.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == target:
            return path
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(_FORMATTER)
    log.addHandler(file_handler)
    return path

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