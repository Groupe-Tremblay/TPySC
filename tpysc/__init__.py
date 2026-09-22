import logging
import time

from .mesh import Mesh2D
from .tpsc import Tpsc
from .tpscplus import TpscPlus
from .utils import pade

logging.getLogger(__name__).addHandler(logging.NullHandler())


def enable_console_logging(level: int = logging.INFO) -> None:
    """
    Attach a simple console handler to the tpysc logger hierarchy.
    Convenient for interactive or notebook use. Safe to call more than once.

    :param level: Minimum severity to display.
    :type level: int
    """
    pkg_logger = logging.getLogger("tpysc")

    if not any(isinstance(h, logging.StreamHandler) for h in pkg_logger.handlers):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # centiseconds instead of the default 3-digit milliseconds
        formatter.formatTime = lambda record, datefmt=None: (
            f"{time.strftime(datefmt, time.localtime(record.created))}"
            f".{int(record.msecs / 10):02d}"
        )
        handler.setFormatter(formatter)
        pkg_logger.addHandler(handler)

    pkg_logger.setLevel(level)
