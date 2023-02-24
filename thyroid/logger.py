import logging 
from datetime import datetime 
import os, sys 

#log file name 
LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y__%I_%M_%S')}.log"

# directory name 
LOG_FILE_DIR = os.path.join(os.getcwd(),"logs")

# create folder if not exists
os.makedirs(LOG_FILE_DIR, exist_ok=True)

# Log file path 
LOG_FILE_PATH = os.path.join(LOG_FILE_DIR, LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH, 
    format="[ %(asctime)s ] %(filename)s: %(lineno)d %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)

