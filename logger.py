# ----------------------------------------------Import required Modules----------------------------------------------- #

import datetime
import logging

# ------------------------------------------------Log status Function------------------------------------------------- #

logging.basicConfig(filename='train.log',
                    filemode='w',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.DEBUG)

def log_status(logLevel, msg):
    """
    Simple function to print log messages to console. \n
    Three types: 1-ERROR, 2-WARN and 3-INFO\n
    :param logLevel: Integer representing the log level\n
    :param msg: the log message to display\n
    :return: prints out the full log message with log level and corresponding message
    """
    logLevelString = {1: "[Error]", 2:"[WARN]", 3:"[INFO]"}

    now = datetime.datetime.now()

    log_msg = "{} {}:- {}".format(logLevelString[logLevel], now.strftime("%d-%m-%Y %H:%M:%S%f"), msg)
    print(log_msg)
    logging.info(log_msg)