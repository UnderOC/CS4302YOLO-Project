# CS4302YOLO-Project
This is the repository for the final porject of CS4302, 2024 fall.


# Environment Setup
System:
- Ubuntu: 20.04.1   
- GCC: 9.4.0
- GPU Driver Version: 515.48.07  Driver CUDA Version: 11.7    

Create python environment
```bash
conda create -n yolo python=3.8
conda activate yolo
conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
conda install -c pytorch magma-cuda110

# Install cuda >=10.2, as required by pytorch
conda install -c conda-forge cudatoolkit=11.6
conda install cuda cuda-nvcc -c nvidia/label/cuda-11.6.0
    # $ nvcc -V
        # nvcc: NVIDIA (R) Cuda compiler driver
        # Copyright (c) 2005-2021 NVIDIA Corporation
        # Built on Fri_Dec_17_18:16:03_PST_2021
        # Cuda compilation tools, release 11.6, V11.6.55
        # Build cuda_11.6.r11.6/compiler.30794723_0
```

Compile pytorch
```bash

make clean

# export CUDA_HOME=$CONDA_PREFIX
# export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LDFLAGS="-L$CONDA_PREFIX/lib $LDFLAGS"

export CMAKE_LIBRARY_PATH=$CONDA_PREFIX/lib:$CMAKE_LIBRARY_PATH
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

cd pytorch
python setup.py install

```

Troubleshooting: The following commands may help when you encounter errors.
- `export CUDA_NVCC_EXECUTABLE=$(which nvcc)`
- `conda install libcusparse-dev -c nvidia/label/cuda-11.6.0` `conda install libcusolver-dev -c nvidia/label/cuda-11.6.0` 

Check correctness
```bash
cd .. # don't run python in the pytorch directory, due to name conflict of 'torch'


$ python
Python 3.8.20 (default, Oct  3 2024, 15:24:27)
[GCC 11.2.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> A = torch.
KeyboardInterrupt
>>> torch.cuda.is_available()
True
>>> torch.cuda.device_count()
8
>>> A = torch.arange(18).reshape(3,6)
>>> A
tensor([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11],
        [12, 13, 14, 15, 16, 17]])
>>> A*A
tensor([[  0,   1,   4,   9,  16,  25],
        [ 36,  49,  64,  81, 100, 121],
        [144, 169, 196, 225, 256, 289]])
>>> A.cuda()*A.cuda()
tensor([[  0,   1,   4,   9,  16,  25],
        [ 36,  49,  64,  81, 100, 121],
        [144, 169, 196, 225, 256, 289]], device='cuda:0')
>>> print(torch.cuda.get_device_name(0))
NVIDIA GeForce RTX 3080 Ti
```
