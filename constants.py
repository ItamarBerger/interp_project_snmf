import os
from enum import StrEnum

LOG_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging.yaml")
LOGS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

class LogColor(StrEnum):
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"