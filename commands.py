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
# tensorboard --logdir=logs/gradient_tape

