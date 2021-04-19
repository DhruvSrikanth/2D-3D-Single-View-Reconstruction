# Commands used in 3D reconstruction project

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# Python version should be 3.6-3.8
# Install virtual environment -
# Create inside Project folder
# sudo apt install python3-venv
# python3 -m venv 3dr_venv

# Activate virtual environment -
# Command must be used in Project folder
# source 3dr_venv/bin/activate
# sudo pip install --user tensorflow-gpu==2.4.0 (run after activating venv)

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# Unpack Datasets .tgz files with -
# tar xzvf your-file-name.tgz
# tar -xzf your-file-name.tgz

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# Some Helpful Github commands -
# Initialize git repo <local-dir>
# git init

# Add files
# git add <file-extension>

# Reset Changes Before Commit
# git reset

# Checkout To Different Branches
# git checkout <branch_name>

# Merge Two Branches
# git checkout <branch_to_merge_into>
# git merge <branch_to_merge_with>

# Commit Changes To Local Git Repo
# git commit -m "Message"

# Push Changes To Global Git Repo
# git push <remote_name> <local_branch_name>:<remote_branch_name>

# Pull Codes From Global Git Repo To Local Repo
# git pull origin master
# git pull <remote_name> <remote_branch_name>:<local_branch_name>

# Delete Local Branch
# git branch -d <branch_name>

# Fix non fast forward error
# git fetch origin <remote_branch>
# git merge origin <remote_branch>
# or
# git pull --rebase origin master

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# Tensorboard
# tensorboard --logdir=logs/gradient_tape --host 0.0.0.0 (VM)
# ssh drs@20.193.228.25 -N -f -L localhost:16006:localhost:6006 (local)

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# tmux usage -
# create - tmux new -s name
# attach - tmux attach -t name
# view - tmux ls
# detach - ctrl+B D

# ---------------------------CUDA Installation commands ubuntu 18.04 lts------------------------------------------#
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
# sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_amd64.deb
# sudo apt-key add /var/cuda-repo-ubuntu1804-11-0-local/7fa2af80.pub
# sudo apt-get update
# sudo apt-get -y install cuda

# --------------------------------------cuDNN installation commands -------------------------------------------------#
# wget https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/8.0.4/11.0_20200923/cudnn-11.0-linux-x64-v8.0.4.30.tgz
# tar -xzvf cudnn-11.0-linux-x64-v8.0.4.30.tgz
# sudo cp cuda/include/cudnn*.h /usr/local/cuda/include 
# sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 
# sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

# to run shell script with bad interpreter error, run this command before
# sed -i -e 's/\r$//' init_config.sh
# sed -i -e 's/\r$//' install.sh

# on vim to convert from dos to unix line endings use ":set ff=unix" and then save file
# to get back dos line endings use ":set ff=dos" and then save file