import argparse
import numpy as np
import tensorflow as tf

@tf.RegisterGradient("Add2")
def add2_grad(op, *grads):
    input_tensor = op.inputs[0]
    output_grad = grads[0]
    return cuda_module.time_two_grad(input_tensor, output_grad)


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
        raise NotImplementedError
    elif args.compiler == 'setup':
        raise NotImplementedError
    elif args.compiler == 'cmake':
        cuda_module = tf.load_op_library('build/libadd2.so')
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
