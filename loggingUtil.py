import logging
from configparser import ConfigParser
from sys import stdout

from tensorflow.python.platform import tf_logging

config = ConfigParser()
config.read('./config/config.ini')
logging_configs = config['Logging']

def get_new_logger(name) -> logging.getLoggerClass():
    """Returns a new logger

    :param name: Name of the logger (recommended to be name of the class that requests a logger)
    :return: Logger
    """
    logger = logging.getLogger(name)
    handler = logging.StreamHandler(stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.parent = None
    try:
        if logging_configs[name] == 'DEBUG':
            logger.setLevel(logging.DEBUG)
        elif logging_configs[name] == 'INFO':
            logger.setLevel(logging.INFO)
        elif logging_configs[name] == 'WARNING':
            logger.setLevel(logging.WARNING)
        elif logging_configs[name] == 'ERROR':
            logger.setLevel(logging.ERROR)
        elif logging_configs[name] == 'FATAL':
            logger.setLevel(logging.FATAL)
    except KeyError as e:
        print(f"No logging level is specified for {name}, please specify one in "
              f"../config/config.ini in Logging. Logging level will be set to WARNING")
        logger.setLevel(logging.WARNING)
    return logger

tf_logging._logger = get_new_logger('tensorflow')
