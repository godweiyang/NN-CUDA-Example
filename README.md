# Neural Network CUDA Example
A simple example for neural network toolkits (PyTorch, TensorFlow, etc.) calling custom CUDA operators.

We provide three ways to compile the CUDA kernel and its cpp warpper, including jit, setuptools and cmake.

## Environments
* NVIDIA Driver: 418.116.00
* CUDA: 11.0
* Python: 3.7.3
* PyTorch: 1.7.0+cu110
* TensorFlow: 2.4.1
* CMake: 3.16.3
* Ninja: 1.10.0
* GCC: 8.3.0

*Cannot ensure successful running in other environments.*

## Code structure
```shell
├── include
│   └── add2.h # header file of add2 cuda kernel
├── kernel
│   └── add2_kernel.cu # add2 cuda kernel
├── pytorch
│   ├── add2_ops.cpp # torch warpper of add2 cuda kernel
│   ├── time.py # time comparison of cuda kernel and torch
│   ├── train.py # training using custom cuda kernel
│   ├── setup.py
│   └── CMakeLists.txt
├── tensorflow
│   ├── add2_ops.cpp # tensorflow warpper of add2 cuda kernel
│   ├── time.py # time comparison of cuda kernel and tensorflow
│   ├── train.py # training using custom cuda kernel
│   └── CMakeLists.txt
├── LICENSE
└── README.md
```

## Compile cpp and cuda
### JIT
**PyTorch**  
Directly run python code as in next section.

**TensorFlow**  
Not implemented.

### Setuptools
**PyTorch**  
```shell
python3 setup.py install
```

**TensorFlow**  
Not implemented.

### CMake
**PyTorch**  
```shell
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ../pytorch
make
```

**TensorFlow**  
```shell
mkdir build
cd build
cmake  ../tensorflow
make
```

## Run python
`$CODEBASE` in the following represents `pytorch` or `tensorflow`.

*Note that for TensorFlow, the compiler only supports cmake currently.*

### Compare kernel time
*Note that for TensorFlow, the time statistics is not correct currently.*

```shell
python3 $CODEBASE/time.py --compiler jit
python3 $CODEBASE/time.py --compiler setup
python3 $CODEBASE/time.py --compiler cmake
```

### Train model
```shell
python3 $CODEBASE/train.py --compiler jit
python3 $CODEBASE/train.py --compiler setup
python3 $CODEBASE/train.py --compiler cmake
```

## Details (in Chinese)
[https://godweiyang.com/2021/03/18/torch-cpp-cuda](https://godweiyang.com/2021/03/18/torch-cpp-cuda)  
[https://godweiyang.com/2021/03/18/torch-cpp-cuda-2](https://godweiyang.com/2021/03/18/torch-cpp-cuda-2)  
[https://godweiyang.com/2021/03/18/torch-cpp-cuda-3](https://godweiyang.com/2021/03/18/torch-cpp-cuda-3)

## F.A.Q
Coming soon...