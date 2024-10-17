import logging
import logging.handlers
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

file_loc = ROOT / "./logs/app.log"


def get_logger(name):

    logger = logging.getLogger(name)

    log_level = 'DEBUG'
    if log_level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif log_level == "INFO":
        logger.setLevel(logging.INFO)
    elif log_level == "WARNING":
        logger.setLevel(logging.WARNING)
    elif log_level == "ERROR":
        logger.setLevel(logging.ERROR)
    elif log_level == "CRITICAL":
        logger.setLevel(logging.CRITICAL)
    else:
        logger.setLevel(logging.INFO)

    handler_file = RotatingFileHandler(
        file_loc, mode = 'a', maxBytes = 1024 * 1024, backupCount = 5,
        encoding = None, delay = False
    )

    log_formatter = logging.Formatter(
        '[%(asctime)s] - %(name)s - {%(filename)s:%(lineno)d} - %(levelname)s - %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )

    handler_file.setFormatter(log_formatter)

    if log_level == "DEBUG":
        handler_file.setLevel(logging.DEBUG)
    elif log_level == "INFO":
        handler_file.setLevel(logging.INFO)
    elif log_level == "WARNING":
        handler_file.setLevel(logging.WARNING)
    elif log_level == "ERROR":
        handler_file.setLevel(logging.ERROR)
    elif log_level == "CRITICAL":
        handler_file.setLevel(logging.CRITICAL)
    else:
        handler_file.setLevel(logging.INFO)

    logger.addHandler(handler_file)
    return logger
