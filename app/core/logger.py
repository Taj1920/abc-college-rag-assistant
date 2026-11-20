import logging
import os

os.makedirs("./logs",exist_ok=True)
def get_logger(name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        handlers=[logging.FileHandler(filename="./logs/app.log"),
                  logging.StreamHandler()]
    )
    return logging.getLogger(name)

if __name__=="__main__":
    logger = get_logger(__name__)
    logger.info("Test log message")