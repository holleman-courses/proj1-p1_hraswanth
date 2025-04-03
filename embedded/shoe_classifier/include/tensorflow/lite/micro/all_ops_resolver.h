#ifndef TENSORFLOW_LITE_MICRO_ALL_OPS_RESOLVER_H_
#define TENSORFLOW_LITE_MICRO_ALL_OPS_RESOLVER_H_

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

namespace tflite {

class AllOpsResolver : public MicroMutableOpResolver<10> {
 public:
  AllOpsResolver() {
    AddConv2D();
    AddDepthwiseConv2D();
    AddAveragePool2D();
    AddMaxPool2D();
    AddFullyConnected();
    AddSoftmax();
    AddReshape();
    AddQuantize();
    AddDequantize();
  }

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_ALL_OPS_RESOLVER_H_
