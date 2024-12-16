#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Initialize conda for bash
echo "Initializing Conda..."
# This ensures that the 'conda' command is available
# Adjust the path below if Conda is installed in a different location
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create the conda environment with Python 3.8
echo "Creating conda environment 'yolo' with Python 3.8..."
conda create -n yolov5 python=3.8

# Activate the newly created environment
echo "Activating conda environment 'yolo'..."
conda activate yolov5

# Install the required packages
echo "Installing required packages..."
conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

# Install magma-cuda110 from the pytorch channel
echo "Installing magma-cuda110 from pytorch channel..."
conda install -c pytorch magma-cuda110

# Install cudatoolkit version 11.6 from conda-forge
# echo "Installing cudatoolkit from conda-forge..."
conda install -c conda-forge cudatoolkit=11.8

# Install cuda and cuda-nvcc from NVIDIA's CUDA 11.6 label
echo "Installing cuda and cuda-nvcc..."
conda install -c nvidia/label/cuda-11.8.0 cuda cuda-nvcc
conda install libcusparse-dev -c nvidia/label/cuda-11.8.0
conda install libcusolver-dev -c nvidia/label/cuda-11.8.0

# Install specific versions of gcc and g++ from conda-forge
echo "Installing gcc and g++ from conda-forge..."
conda install -c conda-forge gxx_linux-64=9.4 gcc_impl_linux-64=9.4 gcc_linux-64=9.4

# Set CC and CXX environment variables to the conda-provided gcc and g++
echo "Setting CC and CXX environment variables..."
export CC=$(which x86_64-conda-linux-gnu-gcc)
export CXX=$(which x86_64-conda-linux-gnu-g++)

# Create symbolic links for gcc and g++ in the conda environment's bin directory
echo "Creating symbolic links for gcc and g++..."
ln -sf $(which x86_64-conda-linux-gnu-gcc) $CONDA_PREFIX/bin/gcc
ln -sf $(which x86_64-conda-linux-gnu-g++) $CONDA_PREFIX/bin/g++

# Display the paths of gcc and g++
echo "Verifying package locations..."
which gcc
which g++

# Set additional environment variables
echo "Setting additional environment variables..."
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LDFLAGS="-L$CONDA_PREFIX/lib $LDFLAGS"
export CUDA_NVCC_EXECUTABLE=$(which nvcc)

export CMAKE_LIBRARY_PATH=$CONDA_PREFIX/lib:$CMAKE_LIBRARY_PATH
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

export TRACE_KERNEL=1
export USE_CUDNN=0

echo "Conda environment 'yolo' setup completed successfully."
