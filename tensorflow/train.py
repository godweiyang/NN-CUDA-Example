import os
import argparse
import numpy as np
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

@tf.RegisterGradient("Add2")
def add2_grad(op, *grads):
    grad_output = grads[0]
    return grad_output, grad_output


class AddModel(tf.keras.Model):
    def __init__(self, n):
        super(AddModel, self).__init__()
        self.n = n

    def build(self, input_shape):
        self.a = self.add_weight(name="a",
                                 shape=(self.n,),
                                 trainable=True,
                                 initializer=tf.random_normal_initializer(
                                     mean=0., stddev=1.0))
        self.b = self.add_weight(name="b",
                                 shape=(self.n,),
                                 trainable=True,
                                 initializer=tf.random_normal_initializer(
                                     mean=0., stddev=1.0))
        super(AddModel, self).build(input_shape)

    def call(self, tmp=None):
        a2 = tf.math.square(self.a)
        b2 = tf.math.square(self.b)
        c = cuda_module.add2(a2, b2)
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
    opt = tf.keras.optimizers.Adam(0.1)
    loss = tf.keras.losses.MeanAbsoluteError()

    print("Configuring model...")
    model.compile(optimizer=opt,
                  loss=loss)
    
    print("Begin training...")
    model.fit(x=np.array([1.]),
              y=np.array([0.]),
              batch_size=1,
              epochs=100)
