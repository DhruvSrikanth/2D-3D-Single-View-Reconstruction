# ----------------------------------------------Import required Modules----------------------------------------------- #

import logging
import sys

# ------------------------------------------------Log status Function------------------------------------------------- #

sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
sh.setFormatter(formatter)

logger_train = logging.getLogger('')
logger_train.setLevel(logging.INFO)
fh_train = logging.FileHandler('train.log')
fh_train.setFormatter(formatter)
logger_train.addHandler(fh_train)
logger_train.addHandler(sh)

logger_test = logging.getLogger('')
logger_test.setLevel(logging.INFO)
fh_test = logging.FileHandler('test.log')
fh_test.setFormatter(formatter)
logger_test.addHandler(fh_test)
logger_test.addHandler(sh)