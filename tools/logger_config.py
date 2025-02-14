import logging

# Create a logger
logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)  # Set the logging level (DEBUG, INFO, etc.)

# Create a file handler (logs to a file)
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.DEBUG)  # Log level for file

# Create a console handler (logs to the terminal)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Log level for console

# Define a common log format
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)

# Apply the format to handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger (file + console)
logger.addHandler(file_handler)
logger.addHandler(console_handler)