
import logging

def setup_logger(logger_name: str) -> logging.Logger:
    """
    Set up a logger with the specified name.
    """
    logger = logging.getLogger(logger_name)
    if not logger.hasHandlers():
        handler = logging.FileHandler(logger_name)
        logger.addHandler(handler)
        logger.propagate = False
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.info("Logger initialized")
    else:
        logger.info("Logger already initialized")
    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)