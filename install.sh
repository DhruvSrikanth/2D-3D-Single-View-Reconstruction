#!/bin/bash
## This gist contains instructions about cuda v10.1 and cudnn 7.6 installation in Ubuntu 18.04 for Tensorflow 2.1.0

### steps ####
# verify the system has a cuda-capable gpu
# download and install the nvidia cuda toolkit and cudnn
# setup environmental variables
# verify the installation
###

### If you have previous installation remove it first. 
 sudo apt-get purge nvidia*
 sudo apt remove nvidia-*
 sudo rm /etc/apt/sources.list.d/cuda*
 sudo apt-get autoremove && sudo apt-get autoclean
 sudo rm -rf /usr/local/cuda*


### to verify your gpu is cuda enable check
lspci | grep -i nvidia

### gcc compiler is required for development using the cuda toolkit. to verify the version of gcc install enter
gcc --version

# system update
sudo apt-get update
sudo apt-get upgrade


# install other import packages
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev


# # first get the PPA repository driver
# sudo add-apt-repository ppa:graphics-drivers/ppa
# sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
# echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list
# sudo apt-get update

#  # installing CUDA-10.1
# sudo apt-get -o Dpkg::Options::="--force-overwrite" install cuda-11-0 cuda-drivers

# these commands are for ubuntu 18.04,
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu1804-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda-11.0

# for ubuntu 20.04 (college system)
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu2004-11-0-local_11.0.2-450.51.05-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu2004-11-0-local_11.0.2-450.51.05-1_amd64.deb
# sudo apt-key add /var/cuda-repo-ubuntu2004-11-0-local/7fa2af80.pub
# sudo apt-get update
# sudo apt-get -y install cuda-11.0

# driver install
# sudo apt-get install cuda-drivers-452


# setup your paths
echo 'export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig

# install cuDNN v7.6
# in order to download cuDNN you have to be regeistered here https://developer.nvidia.com/developer-program/signup
# then download cuDNN v7.6 form https://developer.nvidia.com/cudnn

CUDNN_TAR_FILE="cudnn-11.0-linux-x64-v8.0.4.30.tgz"
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.4/11.0_20200923/cudnn-11.0-linux-x64-v8.0.4.30.tgz
tar -xzvf ${CUDNN_TAR_FILE}


# copy the following files into the cuda toolkit directory.
sudo cp -P cuda/include/cudnn*.h /usr/local/cuda-11.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.0/lib64/
sudo chmod a+r /usr/local/cuda-11.0/include/cudnn*.h /usr/local/cuda-11.0/lib64/libcudnn*

# Finally, to verify the installation, check
nvidia-smi
nvcc -V

# install Tensorflow (an open source machine learning framework)
# I choose version 2.1.0 because it is stable and compatible with CUDA 10.1 Toolkit and cuDNN 7.6

# sudo pip3 install --user tensorflow-gpu==2.4.0