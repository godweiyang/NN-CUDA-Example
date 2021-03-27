import time
import argparse
import numpy as np
import torch

# c = a + b (shape: [n])
n = 1024 * 1024
a = torch.rand(n, device="cuda:0")
b = torch.rand(n, device="cuda:0")
cuda_c = torch.rand(n, device="cuda:0")

ntest = 10

def show_time(func):
    times = list()
    res = None
    # GPU warm up
    for _ in range(10):
        res = func()
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        func()
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()
        times.append((end_time-start_time)*1e6)
    return times, res

def run_cuda():
    if args.compiler == 'jit':
        cuda_module.torch_launch_add2(cuda_c, a, b, n)
    elif args.compiler == 'setup':
        add2.torch_launch_add2(cuda_c, a, b, n)
    elif args.compiler == 'cmake':
        torch.ops.add2.torch_launch_add2(cuda_c, a, b, n)
    else:
        raise Exception("Type of cuda compiler must be one of jit/setup/cmake.")

    return cuda_c

def run_torch():
    c = a + b
    return c.contiguous()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', type=str, choices=['jit', 'setup', 'cmake'], default='jit')
    args = parser.parse_args()

    if args.compiler == 'jit':
        from torch.utils.cpp_extension import load
        cuda_module = load(name="add2",
                           extra_include_paths=["include"],
                           sources=["pytorch/add2_ops.cpp", "kernel/add2_kernel.cu"],
                           verbose=True)
    elif args.compiler == 'setup':
        import add2
    elif args.compiler == 'cmake':
        torch.ops.load_library("build/libadd2.so")
    else:
        raise Exception("Type of cuda compiler must be one of jit/setup/cmake.")

    print("Running cuda...")
    cuda_time, cuda_res = show_time(run_cuda)
    print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

    print("Running torch...")
    torch_time, torch_res = show_time(run_torch)
    print("Torch time:  {:.3f}us".format(np.mean(torch_time)))

    torch.allclose(cuda_res, torch_res)
    print("Kernel test passed.")
