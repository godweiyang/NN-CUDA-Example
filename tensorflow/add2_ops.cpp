#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "add2.h"

using namespace tensorflow;

class Add2Op: public OpKernel {
    public:
        explicit Add2Op(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& a = context->input(0);
        const Tensor& b = context->input(1);
        Tensor* c = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0,
                                                         a.shape(),
                                                         &c));
        launch_add2(c->flat<float>().data(),
                    a.flat<float>().data(),
                    b.flat<float>().data(),
                    static_cast<int>(a.NumElements()));
    }
};

REGISTER_OP("Add2")
    .Input("a: float")
    .Input("b: float")
    .Output("c: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("Add2").Device(DEVICE_GPU), Add2Op);