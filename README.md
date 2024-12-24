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

Set additional environment variables (Use `echo $` to check the variables)
```bash
conda activate yolov5
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LDFLAGS="-L$CONDA_PREFIX/lib $LDFLAGS"
export CUDA_NVCC_EXECUTABLE=$(which nvcc)

export CMAKE_LIBRARY_PATH=$CONDA_PREFIX/lib:$CMAKE_LIBRARY_PATH
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

export TRACE_KERNEL=1
export USE_CUDNN=0
export DEBUG=1
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

## Run with Profiler and Kernel Count
Modify `python/tools/kernel_count.py` to output the use of kernels in decending order:
```python
def main():
    # Execute the provided Python script and redirect output to run.log
    cmd = 'python ' + sys.argv[1] + ' > run.log'
    os.system(cmd)
    
    kernel_count = {}
    
    # Parse run.log to count kernel occurrences
    with open('run.log', encoding='utf-8') as log_file:
        for line in log_file:
            info_idx = line.find("$dispatch kernel")
            if info_idx != -1:
                line = line[info_idx:]
                parts = line.split()
                if len(parts) >= 3:
                    kernel_name = parts[2]
                    if kernel_name in kernel_count:
                        kernel_count[kernel_name] += 1
                    else:
                        kernel_count[kernel_name] = 1
    
    # Sort kernel counts by descending order and write to ordered_kernel.txt
    sorted_kernels = sorted(kernel_count.items(), key=lambda x: x[1], reverse=True)
    with open('ordered_kernel.txt', 'w', encoding='utf-8') as ord_file:
        for name, count in sorted_kernels:
            ord_file.write(f"{name} : {count}\n")
```

Import `torch.autograd.profiler` to measure the time and memory consumption of the model’s operators, see `yolov5/val.py`.
```python
import torch.autograd.profiler as profiler

# in function run
with profiler.profile(with_stack=True, profile_memory=True) as prof:
        for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
                ...
print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=20))
```

Run the valuation process:
```bash
python pytorch/tools/kernel_count/kernel_count.py yolov5/val.py --weights yolov5s.pt --data coco128.yaml --img 640
```

The output of kernel_count is in `ordered_kernel.txt`:
```
"copy_" : 6150
"fill_out" : 3889
"ufunc_add_CUDA" : 3596
"index_cuda" : 3518
"_local_scalar_dense_cuda" : 3424
"copy_kernel" : 2144
"nonzero_cuda" : 1896
"div_true_cuda" : 1640
"cat_cuda" : 1615
"clamp_min_scalar_cuda" : 1268
"ge_cuda" : 1260
"bitwise_and_cuda" : 1260
"prod_cuda" : 754
"add_stub" : 617
"fill_cpu" : 606
"_local_scalar_dense_cpu" : 567
"check_convert" : 511
"index_put" : 508
"mul_cuda" : 402
"gt_cuda" : 384
"slow_conv2d_cuda" : 336
"silu_cuda" : 319
"eq_cuda" : 254
"min_elementwise_cuda" : 252
"max_elementwise_cuda" : 252
"fill_cuda" : 249
"div_cpu" : 193
"sort" : 170
"index_select_cuda" : 128
"index_select_out_cuda_impl" : 128
"arange_cuda" : 120
"check_uniform_bounds" : 114
"uniform_kernel_cpu" : 114
"sqrt_cuda" : 114
"addmm_cuda" : 114
...
```

The output of profiler is in `run.log`:
```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------------------------------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  Source Location                                                              
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------------------------------------  
enumerate(DataLoader)#_MultiProcessingDataLoaderIter...         6.48%     207.151ms         6.48%     207.161ms     207.161ms          -4 b        -292 b           0 b           0 b             1  yolov5/val.py(586): main                                                     
                                                                                                                                                                                                     site-packages/torch/autograd/grad_mode.py(27): decorate_context              
                                                                                                                                                                                                     yolov5/val.py(331): run                                                      
                                                                                                                                                                                                     site-packages/tqdm/std.py(1160): __iter__                                    
                                                                                                                                                                                                     utils/dataloaders.py(240): __iter__                                          
                                                                                                                                                                                                                                                                                  
                                            aten::copy_         4.12%     131.685ms         4.12%     131.685ms       2.058ms           0 b           0 b           0 b           0 b            64  yolov5/val.py(586): main                                                     
                                                                                                                                                                                                     site-packages/torch/autograd/grad_mode.py(27): decorate_context              
                                                                                                                                                                                                     yolov5/val.py(331): run                                                      
                                                                                                                                                                                                     utils/plots.py(139): output_to_target                                        
                                                                                                                                                                                                     <built-in method __enter__ of _thread.lock object at 0x7f5791e68270>         
                                                                                                                                                                                                                                                                                  
                             aten::_slow_conv2d_forward         4.12%     131.534ms         7.35%     234.796ms     978.317us           0 b           0 b      11.31 Gb      -1.12 Gb           240  yolov5/val.py(586): main                                                     
                                                                                                                                                                                                     site-packages/torch/autograd/grad_mode.py(27): decorate_context              
                                                                                                                                                                                                     yolov5/val.py(331): run                                                      
                                                                                                                                                                                                     nn.Module: DetectMultiBackend                                                
                                                                                                                                                                                                     models/common.py(679): forward                                               
                                                                                                                                                                                                                                                                                  
                                             aten::sort         3.40%     108.706ms         3.40%     108.779ms     108.779ms           0 b           0 b      27.00 Kb      18.00 Kb             1  yolov5/val.py(586): main                                                     
                                                                                                                                                                                                     site-packages/torch/autograd/grad_mode.py(27): decorate_context              
                                                                                                                                                                                                     yolov5/val.py(331): run                                                      
                                                                                                                                                                                                     utils/general.py(1012): non_max_suppression                                  
                                                                                                                                                                                                     <built-in method acquire of _thread.lock object at 0x7f5791ef03c0>           
                                                                                                                                                                                                                                                                                  
                                            aten::index         3.22%     103.066ms         4.87%     155.657ms     202.678us           0 b           0 b      64.95 Mb      -1.25 Mb           768  yolov5/val.py(586): main                                                     
                                                                                                                                                                                                     site-packages/torch/autograd/grad_mode.py(27): decorate_context              
                                                                                                                                                                                                     yolov5/val.py(331): run                                                      
                                                                                                                                                                                                     utils/general.py(1012): non_max_suppression                                  
                                                                                                                                                                                                                                                                                  
                                          aten::nonzero         3.06%      97.684ms         4.11%     131.310ms     104.214us           0 b           0 b     584.50 Kb           0 b          1260  yolov5/val.py(586): main                                                     
                                                                                                                                                                                                     site-packages/torch/autograd/grad_mode.py(27): decorate_context              
                                                                                                                                                                                                     yolov5/val.py(331): run                                                      
                                                                                                                                                                                                     yolov5/val.py(145): process_batch                                            
                                                                                                                                                                                                     <built-in method where of type object at 0x7f574e84ea60>                     
                                                                                                                                                                                                                                                                                  
                                              aten::cat         3.01%      96.074ms         3.80%     121.325ms     106.425us           0 b           0 b     574.50 Kb     574.50 Kb          1140  yolov5/val.py(586): main                                                     
                                                                                                                                                                                                     site-packages/torch/autograd/grad_mode.py(27): decorate_context              
                                                                                                                                                                                                     yolov5/val.py(331): run                                                      
                                                                                                                                                                                                     yolov5/val.py(145): process_batch                                            
                                                                                                                                                                                                     <built-in method cat of type object at 0x7f574e84ea60>                       
                                                                                                                                                                                                                                                                                  
                                            aten::index         2.73%      87.183ms         3.63%     115.911ms     101.676us           0 b           0 b     570.00 Kb           0 b          1140  yolov5/val.py(586): main                                                     
                                                                                                                                                                                                     site-packages/torch/autograd/grad_mode.py(27): decorate_context              
                                                                                                                                                                                                     yolov5/val.py(331): run                                                      
                                                                                                                                                                                                     yolov5/val.py(145): process_batch                                            
                                                                                                                                                                                                                                                                                  
                              aten::_local_scalar_dense         2.67%      85.325ms         2.67%      85.325ms      24.920us           0 b           0 b           0 b           0 b          3424  yolov5/val.py(586): main                                                     
                                                                                                                                                                                                     site-packages/torch/autograd/grad_mode.py(27): decorate_context              
                                                                                                                                                                                                     yolov5/val.py(331): run                                                      
                                                                                                                                                                                                     utils/metrics.py(134): process_batch                                         
                                                                                                                                                                                                                                                                                  
                                               aten::ge         2.58%      82.571ms         2.58%      82.571ms      65.533us           0 b           0 b       2.60 Mb       2.60 Mb          1260  yolov5/val.py(586): main                                                     
                                                                                                                                                                                                     site-packages/torch/autograd/grad_mode.py(27): decorate_context              
                                                                                                                                                                                                     yolov5/val.py(331): run                                                      
                                                                                                                                                                                                     yolov5/val.py(145): process_batch                                            
                                                                                                                                                                                                                                                                                  
                                            aten::copy_         2.39%      76.280ms         2.39%      76.280ms      60.063us           0 b           0 b           0 b           0 b          1270  yolov5/val.py(586): main                                                     
                                                                                                                                                                                                     site-packages/torch/autograd/grad_mode.py(27): decorate_context              
                                                                                                                                                                                                     yolov5/val.py(331): run                                                      
                                                                                                                                                                                                     utils/general.py(955): scale_boxes                                           
                                                                                                                                                                                                                                                                                  
                                            aten::copy_         2.29%      73.159ms         2.29%      73.159ms     142.889us           0 b           0 b           0 b           0 b           512  yolov5/val.py(586): main                                                     
                                                                                                                                                                                                     site-packages/torch/autograd/grad_mode.py(27): decorate_context              
                                                                                                                                                                                                     yolov5/val.py(331): run                                                      
                                                                                                                                                                                                     utils/general.py(1012): non_max_suppression                                  
                                                                                                                                                                                                     ....13.1a0+bddbd7e-py3.8-linux-x86_64.egg/torchvision/ops/boxes.py(13): nms  
                                                                                                                                                                                                                                                                                  
                                              aten::cat         2.27%      72.579ms         2.27%      72.579ms      63.666us           0 b           0 b     584.50 Kb     584.50 Kb          1140  yolov5/val.py(586): main                                                     
                                                                                                                                                                                                     site-packages/torch/autograd/grad_mode.py(27): decorate_context              
                                                                                                                                                                                                     yolov5/val.py(331): run                                                      
                                                                                                                                                                                                     yolov5/val.py(145): process_batch                                            
                                                                                                                                                                                                     <built-in method stack of type object at 0x7f574e84ea60>                     
                                                                                                                                                                                                                                                                                  
                                      aten::bitwise_and         2.02%      64.663ms         2.02%      64.663ms      51.320us           0 b           0 b       2.60 Mb       2.60 Mb          1260  yolov5/val.py(586): main                                                     
                                                                                                                                                                                                     site-packages/torch/autograd/grad_mode.py(27): decorate_context              
                                                                                                                                                                                                     yolov5/val.py(331): run                                                      
                                                                                                                                                                                                     yolov5/val.py(145): process_batch                                            
                                                                                                                                                                                                                                                                                  
                                            aten::copy_         1.90%      60.873ms         1.90%      60.873ms      20.291ms           0 b           0 b           0 b           0 b             3  yolov5/val.py(586): main                                                     
                                                                                                                                                                                                     site-packages/torch/autograd/grad_mode.py(27): decorate_context              
                                                                                                                                                                                                     yolov5/val.py(331): run                                                      
                                                                                                                                                                                                     utils/plots.py(139): output_to_target                                        
                                                                                                                                                                                                     <built-in method clone of Tensor object at 0x7f5791ee16d0>                   
                                                                                                                                                                                                                                                                                  
                                           aten::select         1.90%      60.579ms         2.71%      86.536ms       5.612us           0 b           0 b           0 b           0 b         15420  yolov5/val.py(586): main                                                     
                                                                                                                                                                                                     site-packages/torch/autograd/grad_mode.py(27): decorate_context              
                                                                                                                                                                                                     yolov5/val.py(331): run                                                      
                                                                                                                                                                                                     nn.Module: DetectMultiBackend                                                
                                                                                                                                                                                                     models/common.py(679): forward                                               
                                                                                                                                                                                                                                                                                  
                                     aten::index_select         1.74%      55.745ms         1.80%      57.374ms     448.234us           0 b           0 b       3.65 Mb           0 b           128  yolov5/val.py(586): main                                                     
                                                                                                                                                                                                     site-packages/torch/autograd/grad_mode.py(27): decorate_context              
                                                                                                                                                                                                     yolov5/val.py(331): run                                                      
                                                                                                                                                                                                     utils/general.py(1012): non_max_suppression                                  
                                                                                                                                                                                                     ....13.1a0+bddbd7e-py3.8-linux-x86_64.egg/torchvision/ops/boxes.py(13): nms  
                                                                                                                                                                                                                                                                                  
                                            aten::copy_         1.52%      48.680ms         1.52%      48.680ms      76.062us           0 b           0 b           0 b           0 b           640  yolov5/val.py(586): main                                                     
                                                                                                                                                                                                     site-packages/torch/autograd/grad_mode.py(27): decorate_context              
                                                                                                                                                                                                     yolov5/val.py(331): run                                                      
                                                                                                                                                                                                     utils/general.py(1012): non_max_suppression                                  
                                                                                                                                                                                                     utils/general.py(885): xywh2xyxy                                             
                                                                                                                                                                                                                                                                                  
                                              aten::add         1.30%      41.634ms         1.30%      41.634ms     162.633us           0 b           0 b       5.50 Mb       5.50 Mb           256  yolov5/val.py(586): main                                                     
                                                                                                                                                                                                     site-packages/torch/autograd/grad_mode.py(27): decorate_context              
                                                                                                                                                                                                     yolov5/val.py(331): run                                                      
                                                                                                                                                                                                     utils/general.py(1012): non_max_suppression                                  
                                                                                                                                                                                                                                                                                  
                                           aten::clamp_         1.28%      41.037ms         1.33%      42.358ms      41.691us           0 b           0 b           0 b           0 b          1016  yolov5/val.py(586): main                                                     
                                                                                                                                                                                                     site-packages/torch/autograd/grad_mode.py(27): decorate_context              
                                                                                                                                                                                                     yolov5/val.py(331): run                                                      
                                                                                                                                                                                                     utils/general.py(955): scale_boxes                                           
                                                                                                                                                                                                     utils/general.py(990): clip_boxes                                            
                                                                                                                                                                                                                                                                                  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------------------------------------  
Self CPU time total: 3.196s
```

Alternative: Use `torch.profiler` to measure the consumption of the operators in CUDA:
```python
from torch.profiler import profile, record_function, ProfilerActivity

# in function run
if torch.cuda.is_available():
    device_prof = 'cuda'
activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
sort_by_keyword = device_prof + "_time_total"
with profile(activities=activities, record_shapes=True) as prof:
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        ...
print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=20))
```

Run the valuation process:
```bash
python pytorch/tools/kernel_count/kernel_count.py yolov5/val.py --weights yolov5s.pt --data coco128.yaml --img 640
```

The output of profiler is in `run.log`:
```
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                 aten::as_strided         4.29%     185.659ms         4.29%     185.659ms       3.566us        2.554s        65.59%        2.554s      49.054us         52070  
                     aten::select        10.33%     446.808ms        12.45%     538.581ms      18.617us  -688540.000us       -17.68%     806.382ms      27.874us         28930  
                aten::convolution         0.09%       3.882ms        15.23%     658.905ms       2.745ms     221.004ms         5.68%     504.186ms       2.101ms           240  
                     aten::conv2d         0.10%       4.540ms        15.33%     663.445ms       2.764ms   -4503.000us        -0.12%     499.683ms       2.082ms           240  
                      aten::silu_         0.13%       5.619ms         0.13%       5.619ms      24.645us     475.705ms        12.22%     475.705ms       2.086ms           228  
                      aten::slice         5.80%     251.099ms         6.90%     298.490ms      25.516us  -198299.000us        -5.09%     378.770ms      32.379us         11698  
          aten::_nnpack_available         0.00%     203.000us         0.00%     203.000us       0.846us     354.374ms         9.10%     354.374ms       1.477ms           240  
                      aten::copy_         8.97%     387.936ms         8.97%     387.936ms      54.242us     332.580ms         8.54%     332.580ms      46.502us          7152  
              aten::empty_strided         1.10%      47.401ms         1.10%      47.401ms      10.327us     330.527ms         8.49%     330.527ms      72.010us          4590  
                         aten::to         1.22%      52.665ms        12.77%     552.648ms     105.831us      41.217ms         1.06%     322.877ms      61.830us          5222  
                      aten::empty         1.45%      62.523ms         1.45%      62.523ms      10.677us     305.403ms         7.84%     305.403ms      52.152us          5856  
               aten::_convolution         0.21%       9.148ms        15.14%     655.023ms       2.729ms  -289448.000us        -7.43%     283.182ms       1.180ms           240  
                   aten::_to_copy         3.89%     168.415ms        11.56%     499.983ms     126.835us  -236907.000us        -6.08%     281.660ms      71.451us          3942  
                aten::bitwise_and         1.16%      50.228ms         1.16%      50.228ms      39.863us     274.207ms         7.04%     274.207ms     217.625us          1260  
                    aten::__and__         0.38%      16.299ms         1.54%      66.527ms      52.799us  -45228.000us        -1.16%     228.979ms     181.729us          1260  
                        aten::cat         5.77%     249.698ms         8.83%     382.039ms     129.024us     116.034ms         2.98%     227.192ms      76.728us          2961  
                aten::thnn_conv2d         0.08%       3.556ms        14.92%     645.672ms       2.690ms     215.832ms         5.54%     218.256ms     909.400us           240  
                      aten::where         1.01%      43.728ms         9.44%     408.256ms     294.557us      76.826ms         1.97%     190.974ms     137.788us          1386  
                         aten::ge         1.16%      50.001ms         1.16%      50.001ms      39.683us     190.393ms         4.89%     190.393ms     151.106us          1260  
                  aten::unsqueeze         1.41%      60.999ms         1.74%      75.125ms      16.417us  -116251.000us        -2.99%     188.219ms      41.132us          4576  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 4.327s
Self CUDA time total: 3.894s
```




# Kernel Survey
1. [`slow_cov2d_cuda`](pytorch/aten/src/ATen/native/cuda/ConvolutionMM2d.cu)


