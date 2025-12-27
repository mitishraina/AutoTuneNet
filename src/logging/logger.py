import logging
import os
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
        "[%(asctime)s] %(lineno)d %(name)s | %(levelname)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setFormatter(formatter)

    log.addHandler(stream_handler)
    log.addHandler(file_handler)

    return log
