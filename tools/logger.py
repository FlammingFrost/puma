import logging
import sys
from datetime import datetime
import os

# Create a logger
logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)  # Set logging level (DEBUG, INFO, etc.)

# Create a file handler (logs to a file)
root_directory = os.path.dirname(os.path.dirname(__file__))
log_directory = f"{root_directory}/logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_filename = f"{log_directory}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)  # Log level for file

# Define a common log format with full file path
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - [%(pathname)s:%(lineno)d] - %(message)s"
)

# Apply format to handlers
file_handler.setFormatter(formatter)

# Remove default console output
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
        logger.removeHandler(handler)

# Add handlers to the logger (file only)
logger.addHandler(file_handler)