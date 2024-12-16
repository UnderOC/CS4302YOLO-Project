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

# make clean

# export CUDA_HOME=$CONDA_PREFIX
# export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LDFLAGS="-L$CONDA_PREFIX/lib $LDFLAGS"

export TRACE_KERNEL=1
export USE_CUDNN=1
export DEBUG=1

export CMAKE_LIBRARY_PATH=$CONDA_PREFIX/lib:$CMAKE_LIBRARY_PATH
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

cd pytorch
python setup.py install

```


Troubleshooting: The following commands may help when you encounter errors.
- `export CUDA_NVCC_EXECUTABLE=$(which nvcc)`
- `conda install libcusparse-dev -c nvidia/label/cuda-11.6.0` `conda install libcusolver-dev -c nvidia/label/cuda-11.6.0` 

If you don't want to compile from source code
```bash
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
```

## Quick Start
You can use `yolo_setup.sh` to create a conda environment. Modify the CUDA version in line 30-36 to fit your gpu version. 
```bash
chmod +x yolo_setup.sh
./yolo_setup.sh
```

Compile torch
```bash
conda activate yolov5
cd pytorch
python setup.py install 
```

Install torchvision
```bash
cd ..
git clone --single-branch -b release/0.13 https://github.com/pytorch/vision
cd vision
python setup.py install
```

Install dependencies for yolov5 (the torch and torchvision depencencies in `requirements.txt` are commented out)
```bash
cd yolov5
pip install -r requirements.txt
```

The folder structure should be like this after the environment setup:
- `CS4302YOLO-Project`
  - `pytorch`
  - `vision`
  - `yolov5`

Troubleshooting: The following commands may help when you encounter errors.
- `export TORCH_CUDA_ARCH_LIST="8.6"`
- `conda install pillow`

## Check correctness
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
>>> print(torch.__version__)
1.12.0a0+git8c4ef23
>>> print(torchvision.__version__)
0.13.1a0+bddbd7e
```

# Run Yolo

```bash
cd yolov5
# It will automatically download weights, dataset, update python library version.
python val.py --weights yolov5s.pt --data coco128.yaml --img 640


# Output looks like:

# val: data=/TinyNAS2024/wjxie/pdc/yolov5/data/coco128.yaml, weights=['yolov5s.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=False, project=runs/val, name=exp, exist_ok=False, half=False, dnn=False
# YOLOv5 🚀 2024-12-10 Python-3.8.20 torch-1.12.0 CUDA:0 (NVIDIA GeForce RTX 3080 Ti, 12054MiB)
# 
# Fusing layers...
# YOLOv5s summary: 213 layers, 7225885 parameters, 0 gradients
# val: Scanning /TinyNAS2024/wjxie/pdc/datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128 [00:                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 4/4 [00:06<00:00,  1.58s/it]
#                    all        128        929      0.712      0.634      0.713      0.475
# Speed: 0.5ms pre-process, 3.1ms inference, 5.6ms NMS per image at shape (32, 3, 640, 640)
# Results saved to runs/val/exp7
```