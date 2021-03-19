# Torch-cuda-example
A simple example for PyTorch calling custom CUDA operators.

## Code structure
```shell
├── include
│   └── add2.h # header file of add2 cuda kernel
├── kernel
│   ├── CMakeLists.txt
│   ├── add2.cu # add2 cuda kernel
│   └── add2.cpp # torch warpper of add2 cuda kernel
├── LICENSE
├── README.md
├── setup.py
├── show_time.py # time comparison of cuda kernel and torch
└── train.py # training using custom cuda kernel
```

## Usage
### Compile cpp and cuda
**JIT**  
Directly run python code.

**CMake (To do)**  
```shell
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ../kernel/
make -j
```

**Setuptools (To do)**  
```shell
python3 setup.py build develop
```

### Run python
**Compare kernel time**  
```shell
python3 show_time.py
```

**Train**  
```shell
python3 train.py
```

## Details
[https://godweiyang.com/2021/03/18/torch-cpp-cuda](https://godweiyang.com/2021/03/18/torch-cpp-cuda)