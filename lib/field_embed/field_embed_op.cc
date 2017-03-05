#include <stdio.h>
#include <cfloat>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("FieldEmbed")
.Attr("limits: list(int)=[]")//field_size+1
.Input("features: int32")//batch_size*field_size
.Input("vals: float")//batch_size*field_size
.Input("weights: float")//features_size*field_size(in)*embed_size
.Input("biases: float")//field_size*field_size(in)*embed_size
.Output("top_data: float")//batch_size*field_size*field_size(in)*embed_size
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        auto batch_size_handle = c->Dim(c->input(0),0);
        auto field_size_handle = c->Dim(c->input(0),1);
        auto embed_size_handle = c->Dim(c->input(2),2);
        std::vector<::tensorflow::shape_inference::DimensionHandle> dims;
        dims.push_back(batch_size_handle);
        dims.push_back(field_size_handle);
        dims.push_back(field_size_handle);
        dims.push_back(embed_size_handle);
        c->set_output(0,c->MakeShape(dims));
        return Status::OK();
        });


REGISTER_OP("FieldEmbedGrad")
.Attr("limits: list(int)=[]")//field_size+1
.Input("features: int32")//batch_size*field_size
.Input("vals: float")//batch_size*field_size
.Input("weights: float")//features_size*field_size(in)*embed_size
.Input("biases: float")//field_size*field_size(in)*embed_size
.Input("grad: float")//batch_size*field_size*field_size(in)*embed_size
.Output("vals_grad: float")
.Output("weights_grad: float")
.Output("biases_grad: float");


class FieldEmbedOp : public OpKernel {
    public:
        explicit FieldEmbedOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context,
                    context->GetAttr("limits", &limits_));
        }
        void Compute(OpKernelContext* context) override
        {
            // Grab the input tensor
            auto limits = limits_;

            const Tensor& features = context->input(0);
            const Tensor& vals = context->input(1);
            const Tensor& weights = context->input(2);
            const Tensor& biases = context->input(3);

            auto features_flat = features.flat<int>();
            auto vals_flat = vals.flat<float>();
            auto weights_flat = weights.flat<float>();
            auto biases_flat = biases.flat<float>();

            int batch_size = features.dim_size(0);
            int field_size = features.dim_size(1);
            int embed_size = weights.dim_size(2);
            int features_size=limits[field_size];

            int dims[4];
            dims[0]=batch_size;
            dims[1]=field_size;
            dims[2]=field_size;
            dims[3]=embed_size;
            TensorShape output_shape;
            TensorShapeUtils::MakeShape(dims, 4, &output_shape);

            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
            auto output = output_tensor->template flat<float>();

            auto shard = [embed_size, field_size, batch_size, &features_flat, &limits, &vals_flat, &weights_flat, &biases_flat, &output](int64 start, int64 limit)
            {
                for(int64 b = start; b < limit; ++b){
                    int n = b;//(batch,field,field_in,embed)
                    int embed = n % embed_size;
                    n /= embed_size;
                    int field_in = n % field_size;
                    n /= field_size;
                    int field = n % field_size;
                    n /= field_size;
                    int batch = n;

                    int feature = *(features_flat.data() + batch * field_size + field) + limits[field];
                    float val = *(vals_flat.data() + batch * field_size + field);
                    float weight = *(weights_flat.data() + (feature * field_size + field_in) * embed_size + embed);
                    float bias = *(biases_flat.data() + (field * field_size + field_in) * embed_size + embed);
                    output(b) = weight * val + bias;
                }
            };
            const DeviceBase::CpuWorkerThreads& worker_threads=
                *(context->device()->tensorflow_cpu_worker_threads());
            const int64 shard_cost=
                features_size*field_size;
            Shard(worker_threads.num_threads,worker_threads.workers,batch_size*field_size*field_size*embed_size,shard_cost,shard);

        }

    private:
        std::vector<int> limits_;
};

class FieldEmbedGradOp : public OpKernel {
    public:
        explicit FieldEmbedGradOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context,
                    context->GetAttr("limits", &limits_));
        }
        void Compute(OpKernelContext* context) override
        {
            auto limits = limits_;
            const Tensor& features = context->input(0);
            const Tensor& vals = context->input(1);
            const Tensor& weights = context->input(2);
            const Tensor& biases = context->input(3);
            const Tensor& grad = context->input(4);

            auto features_flat = features.flat<int>();
            auto vals_flat = vals.flat<float>();
            auto weights_flat = weights.flat<float>();
            auto grad_flat = grad.flat<float>();

            int batch_size = features.dim_size(0);
            int field_size = features.dim_size(1);
            int embed_size = weights.dim_size(2);
            int features_size=limits[field_size];

            Tensor* vals_grad_tensor = NULL;
            Tensor* weights_grad_tensor = NULL;
            Tensor* biases_grad_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, vals.shape(), &vals_grad_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(1, weights.shape(), &weights_grad_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(2, biases.shape(), &biases_grad_tensor));
            auto vals_grad = vals_grad_tensor->template flat<float>();
            auto weights_grad = weights_grad_tensor->template flat<float>();
            auto biases_grad = biases_grad_tensor->template flat<float>();

            int vals_count = batch_size * field_size;
            int weights_count = field_size * field_size * embed_size;
            int biases_count = field_size * field_size * embed_size;


            auto shard = [vals_count, field_size, &features_flat, &limits, embed_size, &weights_flat, &grad_flat, &vals_grad, weights_count, &vals_flat, &weights_grad, &biases_grad, batch_size, biases_count](int64 start, int64 limit){
                for (int64 b = start; b < limit; ++b){
                    int n = b;
                    if(n < vals_count){//vals_grad
                        int field = n % field_size;
                        int batch = n / field_size;
                        int feature = *(features_flat.data() + batch * field_size + field) + limits[field];
                        float val_grad = 0;
                        for(int field_in = 0; field_in < field_size; ++field_in){
                            for(int embed = 0; embed < embed_size; ++embed){
                                float weight = *(weights_flat.data() + (feature * field_size + field_in) * embed_size + embed);
                                float output_grad = *(grad_flat.data() + ((batch * field_size + field) * field_size + field_in) * embed_size + embed);
                                val_grad += weight * output_grad;
                            }
                        }
                        vals_grad(n) = val_grad;
                    }else if(n < vals_count + weights_count){//weights_grad
                        n -= vals_count;
                        int embed = n % embed_size;
                        n /= embed_size;
                        int field_in = n % field_size;
                        n /= field_size;
                        int field = n;
                        for(int feature = limits[field]; feature < limits[field+1]; ++feature){
                            float weight_grad = 0;
                            for(int batch = 0; batch < batch_size; ++batch){
                                float val = *(vals_flat.data() + (batch * field_size + field));
                                float output_grad = *(grad_flat.data() + ((batch * field_size + field) * field_size + field_in) * embed_size + embed);
                                weight_grad += val * output_grad;
                            }
                            weights_grad((feature * field_size + field_in) * embed_size + embed) = weight_grad;
                        }
                    }else{
                        n -= vals_count + weights_count;//biases_grad
                        float bias_grad = 0;
                        for(int batch = 0; batch < batch_size; ++batch){
                            float output_grad = *(grad_flat.data() + batch * biases_count + n);
                            bias_grad += output_grad;
                        }
                        biases_grad(n) = bias_grad;
                    }
                }
            };
            const DeviceBase::CpuWorkerThreads& worker_threads=
                *(context->device()->tensorflow_cpu_worker_threads());
            const int64 shard_cost=
                features_size*field_size;
            Shard(worker_threads.num_threads, worker_threads.workers, vals_count+weights_count+biases_count, shard_cost,shard);
        }
    private:
        std::vector<int> limits_;
};


REGISTER_KERNEL_BUILDER(Name("FieldEmbed").Device(DEVICE_CPU), FieldEmbedOp);
REGISTER_KERNEL_BUILDER(Name("FieldEmbedGrad").Device(DEVICE_CPU), FieldEmbedGradOp);
