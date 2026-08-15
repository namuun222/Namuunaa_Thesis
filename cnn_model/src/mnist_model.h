#ifndef MNIST_MODEL_H
#define MNIST_MODEL_H

#include "model_config.h"
#include "mnist_weights.h"

static const ConvConfig_t mnist_conv_layers[] = {
    { conv0_wt, conv0_bias, 28,  1, 32, 3, 1, 1, CONV0_BIAS_SHIFT, CONV0_OUT_SHIFT, 28 },
    { conv1_wt, conv1_bias, 28, 32, 32, 3, 1, 1, CONV1_BIAS_SHIFT, CONV1_OUT_SHIFT, 28 },
    { conv2_wt, conv2_bias, 14, 32, 64, 3, 1, 1, CONV2_BIAS_SHIFT, CONV2_OUT_SHIFT, 14 },
    { conv3_wt, conv3_bias, 14, 64, 64, 3, 1, 1, CONV3_BIAS_SHIFT, CONV3_OUT_SHIFT, 14 },
};

static const PoolConfig_t mnist_pool_layers[] = {
    { 28, 32, 3, 1, 2, 14 },   // MP1: 28x28x32 -> 14x14x32
    { 14, 64, 5, 0, 3,  4 },   // MP2: 14x14x64 -> 4x4x64
};

static const FCConfig_t mnist_fc_layers[] = {
    { fc1_wt, fc1_bias, 1024, 256, FC1_BIAS_SHIFT, FC1_OUT_SHIFT },
    { fco_wt, fco_bias,  256,  10, FCO_BIAS_SHIFT, FCO_OUT_SHIFT },
};

static const Layer_t mnist_network[] = {
    // type            config                  in out cp  out_size
    { LAYER_CONV,    &mnist_conv_layers[0],   0,  0,  1, 28*28*32 },
    { LAYER_RELU,    NULL,                    0,  0,  0, 28*28*32 },
    { LAYER_CONV,    &mnist_conv_layers[1],   0,  1,  1, 28*28*32 },
    { LAYER_RELU,    NULL,                    1,  1,  0, 28*28*32 },
    { LAYER_POOL,    &mnist_pool_layers[0],   1,  0,  1, 14*14*32 },
    { LAYER_CONV,    &mnist_conv_layers[2],   0,  1,  1, 14*14*64 },
    { LAYER_RELU,    NULL,                    1,  1,  0, 14*14*64 },
    { LAYER_CONV,    &mnist_conv_layers[3],   1,  0,  1, 14*14*64 },
    { LAYER_RELU,    NULL,                    0,  0,  0, 14*14*64 },
    { LAYER_POOL,    &mnist_pool_layers[1],   0,  1,  1,  4*4*64  },
    { LAYER_FC,      &mnist_fc_layers[0],     1,  0,  1,  256     },
    { LAYER_RELU,    NULL,                    0,  0,  0,  256     },
    { LAYER_FC,      &mnist_fc_layers[1],     0,  1,  1,  10      },
    { LAYER_SOFTMAX, NULL,                    1,  1,  0,  10      },
};

static const ModelConfig_t mnist_model = {
    .name         = "MNIST_CNN_v2",
    .layers       = mnist_network,
    .num_layers   = sizeof(mnist_network)/sizeof(mnist_network[0]),
    .num_classes  = 10,
    .input_w      = 28,
    .input_h      = 28,
    .input_ch     = 1,
    .max_act_size = 28*28*32,
};

#endif // MNIST_MODEL_H
