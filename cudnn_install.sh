#!/bin/bash

# cudnn linux file needs to be in same folder as this script

CUDNN_TAR_FILE="cudnn-11.0-linux-x64-v8.0.4.30.tgz"
tar -xzvf ${CUDNN_TAR_FILE}

# copy the following files into the cuda toolkit directory.
sudo cp -P cuda/include/cudnn*.h /usr/local/cuda-11.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.0/lib64/
sudo chmod a+r /usr/local/cuda-11.0/include/cudnn*.h /usr/local/cuda-11.0/lib64/libcudnn*

echo "cudnn install done"

exit 1