import logging

# Configure logging only once
logging.basicConfig(
    filename="app.log",  # Log file
    level=logging.DEBUG,  # Minimum log level
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)

# Create a logger instance
logger = logging.getLogger("app_logger")