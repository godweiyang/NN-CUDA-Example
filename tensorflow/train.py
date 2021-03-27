import argparse
import numpy as np
import torch
from torch import nn
from torch.autograd import Function

class AddModelFunction(Function):
    @staticmethod
    def forward(ctx, a, b, n):
        c = torch.empty(n).to(device="cuda:0")

        if args.compiler == 'jit':
            cuda_module.torch_launch_add2(c, a, b, n)
        elif args.compiler == 'setup':
            add2.torch_launch_add2(c, a, b, n)
        elif args.compiler == 'cmake':
            torch.ops.add2.torch_launch_add2(c, a, b, n)
        else:
            raise Exception("Type of cuda compiler must be one of jit/setup/cmake.")

        return c

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output, grad_output, None)


class AddModel(nn.Module):
    def __init__(self, n):
        super(AddModel, self).__init__()
        self.n = n
        self.a = nn.Parameter(torch.Tensor(self.n))
        self.b = nn.Parameter(torch.Tensor(self.n))
        self.a.data.normal_(mean=0.0, std=1.0)
        self.b.data.normal_(mean=0.0, std=1.0)

    def forward(self):
        a2 = torch.square(self.a)
        b2 = torch.square(self.b)
        c = AddModelFunction.apply(a2, b2, self.n)
        return c

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

    n = 1024

    print("Initializing model...")
    model = AddModel(n)
    model.to(device="cuda:0")

    print("Initializing optimizer...")
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    print("Begin training...")
    for epoch in range(500):
        opt.zero_grad()
        output = model()
        loss = output.sum()
        loss.backward()
        opt.step()
        if epoch % 25 == 0:
            print("epoch {:>3d}: loss = {:>8.3f}".format(epoch, loss))
