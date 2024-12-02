import os
from pathlib import Path
import logging

def get_logger(name: str, file: os.PathLike) -> logging.Logger:
    # Get or create the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Set the overall log level to the lowest level you need

    # Avoid adding duplicate handlers
    if not logger.handlers:
        # File Handler for INFO and higher (excluding DEBUG)
        file_handler = logging.FileHandler(file, mode='w')
        file_handler.setLevel(logging.INFO)  # Handles INFO and higher
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        # Stream Handler for DEBUG and ERROR levels
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)  # Handles DEBUG and ERROR levels
        stream_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(stream_handler)

    # Prevent logs from propagating to the root logger
    logger.propagate = False

    return logger

def remove_old_tf_evenfile(directory:Path):
    if not directory.is_dir():
        return 
    
    old_board_file = list(
        filter(
            lambda x:'events.out.tfevents.' in x.name,
            [_ for _ in directory.iterdir() if _.is_file()]
        )
    )

    for i in old_board_file:
        print(f"remove {i}")
        os.remove(i)