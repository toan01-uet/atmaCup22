import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from logging import INFO, FileHandler, StreamHandler
from pathlib import Path

# Japan Standard Time (JST) timezone
JST = timezone(timedelta(hours=9))


def get_logger(file_name: str, file_dir: Path | str = None) -> logging.Logger:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)

    # Add console handler
    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    logger.addHandler(stream_handler)

    # Use JST timezone for file naming
    jst_time = datetime.now(JST)
    
    # Get the calling script file name
    frame = sys._getframe(1)
    calling_file = frame.f_code.co_filename
    calling_path = Path(calling_file)
    
    # Auto-detect if we're in an experiment
    is_experiment = "experiments" in str(calling_path)
    
    # If file_dir is not specified and we're in an experiment, use the logs folder
    if file_dir is None and is_experiment:
        # Find project root
        project_root = None
        for parent in calling_path.parents:
            if (parent / "experiments").exists() and (parent / "logs").exists():
                project_root = parent
                break
                
        if project_root is None:
            # Fallback to 3 levels up if we can't find the project root
            project_root = calling_path.parent.parent.parent
        
        # Get experiment name from path
        parts = calling_path.parts
        exp_index = parts.index("experiments")
        if exp_index + 1 < len(parts):
            exp_name = parts[exp_index + 1]
        else:
            exp_name = "general"
            
        # Create logs directory for this experiment
        logs_dir = project_root / "logs" / exp_name
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_dir = logs_dir
    elif file_dir is None:
        # For non-experiment files, use current directory if not specified
        file_dir = Path.cwd()
    
    # Create log file with timestamp
    log_file_name = Path(file_dir) / f"{jst_time.strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = FileHandler(log_file_name)
    file_handler.setLevel(INFO)
    
    # Custom formatter with JST timezone
    class JSTFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created, JST)
            if datefmt:
                s = dt.strftime(datefmt)
            else:
                s = dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            return s
    
    formatter = JSTFormatter("[%(asctime)s : %(levelname)s - %(filename)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger