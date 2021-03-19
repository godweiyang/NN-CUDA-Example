import numpy as np
import torch
from torch.utils.cpp_extension import load
from torch import nn
from torch.autograd import Function

cuda_module = load(name="add2",
                   extra_include_paths=["./include"],
                   sources=["./kernel/add2.cpp", "./kernel/add2.cu"],
                   verbose=True)
# torch.ops.load_library("build/libadd2.so")

class AddModelFunction(Function):
    @staticmethod
    def forward(ctx, a, b, n):
        c = torch.empty(n).to(device="cuda:0")
        cuda_module.torch_launch_add2(c, a, b, n)
        # torch.ops.add2.torch_launch_add2(c, a, b, n)
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
    n = 1024
    model = AddModel(n)
    model.to(device="cuda:0")
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(500):
        opt.zero_grad()
        output = model()
        loss = output.sum()
        loss.backward()
        opt.step()
        if epoch % 20 == 0:
            print("epoch: {}, loss={:.3f}".format(epoch, loss))
