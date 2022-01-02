# RUN ONLY ONCE FOR SETUP

# Create a folder containing the 2 dataset tgz files and this shell script.
# Then run the shell script by typing ./init_config.sh or source init_config.sh
# The script will create the dataset folders and extract the tgz files to those folders
# It will then create src folder and git clone the master brance of the repo (needs git username and pwd at this stage)
# Next it will call install.sh script which will install CUDA and cuDNN (the zip file of cuDNN must be downloaded already)
# After this is done, it will create a python virtual env, activate it and run pip on reqirements.txt to install modules

INIT_DIR="$(pwd)"
echo "$INIT_DIR"

DATASET_DIR="datasets/ShapeNet/"
SRC_DIR="src/"
GIT_DIR="3D-Reconstruction"

# dataset files
RENDER_DS_FILE="ShapeNetRendering.tgz"
VOXEL_DS_FILE="ShapeNetVox32.tgz"

# function to change directories
ch_dir () {
  dir_path=$1
  cd $dir_path
}

echo "Dataset directory doesn't exist. Creating now"
mkdir -p "$DATASET_DIR"
echo "Directory created"

# tar -xvzf "$ARG1" -C $DATASET_DIR
tar -xvzf "$RENDER_DS_FILE" -C $DATASET_DIR
tar -xvzf "$VOXEL_DS_FILE" -C $DATASET_DIR
# rm *.tgz

echo "Src code directory doesn't exist. Creating now"
mkdir -p "$SRC_DIR"
echo "Directory created"

ch_dir "$SRC_DIR"
echo "Cloning GitHub repository"
git clone "https://github.com/DhruvSrikanth/3D-Reconstruction.git"

ch_dir ..
sudo mv {"src/${GIT_DIR}"/*,"src/${GIT_DIR}"/.*} "src/"
rmdir "$GIT_DIR"

echo "Done"
exit