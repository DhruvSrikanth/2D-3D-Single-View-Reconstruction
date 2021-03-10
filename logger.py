# ----------------------------------------------Import required Modules----------------------------------------------- #

import logging
import sys

# ------------------------------------------------Log status Function------------------------------------------------- #

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('train.log')
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)