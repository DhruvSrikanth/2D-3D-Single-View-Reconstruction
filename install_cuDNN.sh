# install cuDNN v8.0.4
# in order to download cuDNN you have to be regeistered here https://developer.nvidia.com/developer-program/signup
# then download cuDNN v8.0.4 form https://developer.nvidia.com/cudnn

CUDNN_TAR_FILE="cudnn-11.0-linux-x64-v8.0.4.30.tgz"
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.4/11.0_20200923/cudnn-11.0-linux-x64-v8.0.4.30.tgz
tar -xzvf ${CUDNN_TAR_FILE}


# copy the following files into the cuda toolkit directory.
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-11.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.0/lib64/
sudo chmod a+r /usr/local/cuda-11.0/lib64/libcudnn*

# Finally, to verify the installation, check
nvidia-smi
nvcc -V

echo "cuDNN installation complete"
exit