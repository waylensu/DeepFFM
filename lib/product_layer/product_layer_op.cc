#include <stdio.h>
#include <cfloat>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("ProductLayer")
.Input("bottom_data: float")//batch_size,field_size,field_size(in),embed_size
.Output("top_data: float")//batch_size,(field_size*(field_size+1)/2)
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        auto batch_size_handle = c->Dim(c->input(0),0);
        auto field_size_handle = c->Dim(c->input(0),1);
        ::tensorflow::shape_inference::DimensionHandle mul, add ,divide;
        ::tensorflow::shape_inference::DimensionOrConstant divisor(2);
        c->Multiply(field_size_handle, field_size_handle, &mul);
        c->Add(field_size_handle, mul, &add);
        c->Divide(add, divisor, false, &divide);

        std::vector<::tensorflow::shape_inference::DimensionHandle> dims;
        dims.push_back(batch_size_handle);
        dims.push_back(divide);
        c->set_output(0,c->MakeShape(dims));
        return Status::OK();
        });


REGISTER_OP("ProductLayerGrad")
.Input("bottom_data: float")//batch_size,field_size,field_size(in),embed_size
.Input("top_grad: float")//batch_size,(field_size*(field_size+1)/2)
.Output("bottom_grad: float");//batch_size,field_size,field_size(in),embed_size

class ProductLayerOp: public OpKernel {
    public:
        explicit ProductLayerOp(OpKernelConstruction* context) : OpKernel(context) {
        }
        void Compute(OpKernelContext* context) override
        {

            const Tensor& bottom_data = context->input(0);
            auto bottom_flat = bottom_data.flat<float>();

            int batch_size = bottom_data.dim_size(0);
            int field_size = bottom_data.dim_size(1);
            int embed_size = bottom_data.dim_size(3);

            int dims[2];
            dims[0] = batch_size;
            dims[1] = field_size * (field_size + 1) / 2;

            TensorShape top_shape;
            TensorShapeUtils::MakeShape(dims, 2, &top_shape);

            Tensor* top_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, top_shape, &top_tensor));
            auto top_flat = top_tensor->template flat<float>();

            auto shard = [embed_size, field_size, &bottom_flat, &top_flat, &dims](int64 start, int64 limit)
            {
                for(int b = start; b < limit; ++b){//rhs>=lhs
                    int batch = b / dims[1];
                    int offset = b % dims[1];
                    int n = offset;
                    int lhs;
                    int rhs = field_size;
                    do{
                        lhs = n % rhs;
                        n -= rhs;
                        --rhs;
                    }while(n >= 0);

                    auto lhs_data = bottom_flat.data() + ((batch * field_size + lhs) * field_size + rhs) * embed_size;
                    auto rhs_data = bottom_flat.data() + ((batch * field_size + rhs) * field_size + lhs) * embed_size;
                    float sum = 0;
                    for(int i = 0; i < embed_size; ++i){
                        sum += lhs_data[i] * rhs_data[i];
                    }
                    top_flat(b) = sum;
                }
            };
            const DeviceBase::CpuWorkerThreads& worker_threads=
                *(context->device()->tensorflow_cpu_worker_threads());
            const int64 shard_cost=
                batch_size * field_size;
            Shard(worker_threads.num_threads, worker_threads.workers, dims[0] * dims[1], shard_cost,shard);

        }
};

class ProductLayerGradOp : public OpKernel {
    public:
        explicit ProductLayerGradOp(OpKernelConstruction* context) : OpKernel(context) {
        }
        void Compute(OpKernelContext* context) override
        {

            const Tensor& bottom_data = context->input(0);
            const Tensor& top_grad = context->input(1);
            auto bottom_flat = bottom_data.flat<float>();
            auto top_grad_flat = top_grad.flat<float>();

            int batch_size = bottom_data.dim_size(0);
            int field_size = bottom_data.dim_size(1);
            int embed_size = bottom_data.dim_size(3);
            int top_len = top_grad.dim_size(1);

            TensorShape bottom_grad_shape = bottom_data.shape();
            Tensor* bottom_grad_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, bottom_grad_shape, &bottom_grad_tensor));
            auto bottom_grad_flat = bottom_grad_tensor->template flat<float>();

            auto shard = [embed_size, field_size, &bottom_flat, &bottom_grad_flat, &top_grad_flat, top_len](int64 start, int64 limit)
            {
                for(int64 b = start; b < limit; ++b){
                    int n = b;
                    int embed = n % embed_size;
                    n /= embed_size;
                    int offset = n % top_len;
                    n /= top_len;
                    int batch = n;

                    n = offset;
                    int lhs;
                    int rhs = field_size;
                    do{
                        lhs = n % rhs;
                        n -= rhs;
                        --rhs;
                    }while(n >= 0);
                    float grad = *(top_grad_flat.data() + batch * top_len + offset);
                    int lhs_offset = ((batch * field_size + lhs) * field_size + rhs ) * embed_size + embed;
                    int rhs_offset = ((batch * field_size + rhs) * field_size + lhs ) * embed_size + embed;
                    bottom_grad_flat(lhs_offset) = grad * bottom_flat(rhs_offset);
                    bottom_grad_flat(rhs_offset) = grad * bottom_flat(lhs_offset);
                }
            };
            const DeviceBase::CpuWorkerThreads& worker_threads=
                *(context->device()->tensorflow_cpu_worker_threads());
            const int64 shard_cost=
                batch_size * field_size * field_size * embed_size;
            Shard(worker_threads.num_threads, worker_threads.workers, batch_size * top_len * embed_size, shard_cost, shard);
        }
};

REGISTER_KERNEL_BUILDER(Name("ProductLayer").Device(DEVICE_CPU), ProductLayerOp);
REGISTER_KERNEL_BUILDER(Name("ProductLayerGrad").Device(DEVICE_CPU), ProductLayerGradOp);
