import logging
import sys
from typing import Optional

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure global logging for the whole project (scripts, notebooks).

    - Clears any existing handlers (so we don't get duplicated logs).
    - Installs a single StreamHandler to stdout.
    - Applies a compact formatter with timestamp / level / message.
    - Sets same behavior for terminal and notebook.

    Params:
        level : int (default logging.INFO)
            Minimum log level to display.
    """
    root = logging.getLogger()

    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)

    fmt = "%(asctime)s %(levelname)-5s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    handler.setFormatter(formatter)

    root.addHandler(handler)
    root.setLevel(level)

    root.propagate = False

    for noisy in ("urllib3", "requests", "transformers"):
        lib_logger = logging.getLogger(noisy)
        lib_logger.setLevel(logging.WARNING)
        lib_logger.propagate = True

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Return a module-specific logger.

    Usage in code:
        logger = get_logger(__name__)

    Behavior:
    - Does NOT create new handlers.
    - Just returns a child logger of root.
    - Optionally lets you bump level for a given sub-module.
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)

    logger.propagate = True
    return logger
