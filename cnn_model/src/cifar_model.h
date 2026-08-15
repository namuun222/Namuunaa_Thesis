#ifndef CIFAR_MODEL_H
#define CIFAR_MODEL_H

#include "model_config.h"
#include "cifar_weights.h"

//*****************************************************************************
// Model: ARM's official CMSIS-NN CIFAR-10 example (Caffe-trained,
// pre-quantized q7 by ARM). Input is 32x32x3 RGB, 10 classes:
//   0:airplane 1:automobile 2:bird 3:cat 4:deer
//   5:dog      6:frog       7:horse 8:ship 9:truck
//*****************************************************************************

static const ConvConfig_t cifar_conv_layers[] = {
    // Conv1: 32x32x3 -> 32x32x32 (5x5, pad=2, stride=1, RGB variant)
    { cifar_conv1_wt, cifar_conv1_bias, 32,  3, 32, 5, 2, 1,
      CIFAR_CONV1_BIAS_SHIFT, CIFAR_CONV1_OUT_SHIFT, 32 },
    // Conv2: 16x16x32 -> 16x16x16 (5x5, pad=2, stride=1)
    { cifar_conv2_wt, cifar_conv2_bias, 16, 32, 16, 5, 2, 1,
      CIFAR_CONV2_BIAS_SHIFT, CIFAR_CONV2_OUT_SHIFT, 16 },
    // Conv3: 8x8x16 -> 8x8x32 (5x5, pad=2, stride=1)
    { cifar_conv3_wt, cifar_conv3_bias,  8, 16, 32, 5, 2, 1,
      CIFAR_CONV3_BIAS_SHIFT, CIFAR_CONV3_OUT_SHIFT,  8 },
};

static const PoolConfig_t cifar_pool_layers[] = {
    // Pool1: 32x32x32 -> 16x16x32 (max, 3x3, stride=2)
    { 32, 32, 3, 0, 2, 16 },
    // Pool2: 16x16x16 -> 8x8x16 (max, 3x3, stride=2)
    { 16, 16, 3, 0, 2,  8 },
    // Pool3: 8x8x32 -> 4x4x32 (max, 3x3, stride=2)
    {  8, 32, 3, 0, 2,  4 },
};

static const FCConfig_t cifar_fc_layers[] = {
    // IP1: 512 (=4*4*32) -> 10
    { cifar_ip1_wt, cifar_ip1_bias, 512, 10,
      CIFAR_IP1_BIAS_SHIFT, CIFAR_IP1_OUT_SHIFT },
};

static const Layer_t cifar_network[] = {
    // type            config                    in out cp  out_size
    { LAYER_CONV,    &cifar_conv_layers[0],   0,  0,  1, 32*32*32 },  // Conv1 (RGB)
    { LAYER_RELU,    NULL,                    0,  0,  0, 32*32*32 },  // ReLU
    { LAYER_POOL,    &cifar_pool_layers[0],   0,  1,  1, 16*16*32 },  // Pool1 (max)
    { LAYER_CONV,    &cifar_conv_layers[1],   1,  0,  1, 16*16*16 },  // Conv2
    { LAYER_RELU,    NULL,                    0,  0,  0, 16*16*16 },  // ReLU
    { LAYER_POOL,    &cifar_pool_layers[1],   0,  1,  1,  8*8*16  },  // Pool2 (max)
    { LAYER_CONV,    &cifar_conv_layers[2],   1,  0,  1,  8*8*32  },  // Conv3
    { LAYER_RELU,    NULL,                    0,  0,  0,  8*8*32  },  // ReLU
    { LAYER_POOL,    &cifar_pool_layers[2],   0,  1,  1,  4*4*32  },  // Pool3 (max): act0 -> act1
    { LAYER_FC,      &cifar_fc_layers[0],     1,  0,  1,  10      },  // IP1: act1 -> act0
    { LAYER_SOFTMAX, NULL,                    0,  0,  0,  10      },  // Softmax: act0 in-place
};

static const ModelConfig_t cifar_model = {
    .name         = "CIFAR10_ARM",
    .layers       = cifar_network,
    .num_layers   = sizeof(cifar_network) / sizeof(cifar_network[0]),
    .num_classes  = 10,
    .input_w      = 32,
    .input_h      = 32,
    .input_ch     = 3,
    .max_act_size = 32*32*32,
};

#endif // CIFAR_MODEL_H
