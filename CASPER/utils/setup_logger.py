import os
import logging
def setup_logging(log_file):
    # Create the directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Get a named logger to avoid conflicts
    logger = logging.getLogger("MyAppLogger")

    # Avoid adding duplicate handlers
    if not logger.handlers:
        # Set the overall logging level
        logger.setLevel(logging.INFO)

        # Create a file handler to save logs to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create a stream handler to print logs to console
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        # Define the logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
