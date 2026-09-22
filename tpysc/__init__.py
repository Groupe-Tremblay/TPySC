import logging

from .mesh import Mesh2D
from .tpsc import Tpsc
from .tpscplus import TpscPlus
from .utils import pade

logging.getLogger(__name__).addHandler(logging.NullHandler())


class _CentisecondFormatter(logging.Formatter):
    """
    Formatter using two-digit centiseconds instead of three-digit
    milliseconds, since %(msecs)d cannot be truncated directly in a format
    string.

    :meta private:
    """

    def format(self, record: logging.LogRecord) -> str:
        record.centisecs = int(record.msecs / 10)
        return super().format(record)


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
        handler.setFormatter(
            _CentisecondFormatter(
                "%(asctime)s.%(centisecs)02d - %(name)s - %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        pkg_logger.addHandler(handler)

    pkg_logger.setLevel(level)
