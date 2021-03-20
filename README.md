# Torch-cuda-example
A simple example for PyTorch calling custom CUDA operators.

## Environments
* NVIDIA Driver: 418.116.00
* CUDA: 11.0
* Python: 3.7.3
* PyTorch: 1.7.0+cu110
* CMake: 3.16.3
* Ninja: 1.10.0
* GCC: 8.3.0

*Cannot ensure successful running in other environments.*

## Code structure
```shell
├── include
│   └── add2.h # header file of add2 cuda kernel
├── kernel
│   ├── CMakeLists.txt
│   ├── add2_kernel.cu # add2 cuda kernel
│   └── add2.cpp # torch warpper of add2 cuda kernel
├── LICENSE
├── README.md
├── setup.py
├── time.py # time comparison of cuda kernel and torch
└── train.py # training using custom cuda kernel
```

## Usage
### Compile cpp and cuda
**JIT**  
Directly run python code as in next section.

**CMake (To do)**  
```shell
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ../
make -j
```

**Setuptools**  
```shell
python3 setup.py install
```

### Run python
**Compare kernel time**  
```shell
python3 time.py --compiler jit
python3 time.py --compiler setup
python3 time.py --compiler cmake
```

**Train**  
```shell
python3 train.py --compiler jit
python3 train.py --compiler setup
python3 train.py --compiler cmake
```

## Details
[https://godweiyang.com/2021/03/18/torch-cpp-cuda](https://godweiyang.com/2021/03/18/torch-cpp-cuda)