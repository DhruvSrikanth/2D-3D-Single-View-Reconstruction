#!/bin/bash

# SRC_DIR="src/"

# Uncomment the commnds under this statement to setup virtual environment and install TF and other modules using pip
# Which folder is the virtual env created? (inside src or outside src?)

# ch_dir $SRC_DIR # if virual env should be inside src folder
sudo apt install python3-venv
python3 -m venv 3dr_venv
source 3dr_venv/bin/activate
sudo pip install -r src/requirements.txt