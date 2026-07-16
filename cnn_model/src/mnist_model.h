#ifndef MNIST_MODEL_H
#define MNIST_MODEL_H

#include "model_config.h"
#include "weights.h"  


// Each array holds one entry per conv/pool/fc layer IN THE ORDER THEY
// APPEAR IN THE NETWORK, indexed by the `index` implied by which entry
// of network[] below points to it (network[] entries reference these by
// pointer, e.g. &mnist_conv_layers[0]).
// Field meanings match ConvConfig_t/PoolConfig_t/FCConfig_t defined in
// model_config.h. Comments give the input->output shape for readability.
//*****************************************************************************
static const ConvConfig_t mnist_conv_layers[] = {
    // Conv0: 28x28x1 -> 28x28x32  (3x3 kernel, pad=1, stride=1)
    { conv0_wt, conv0_bias, 28, 1,  32, 3, 1, 1, CONV0_BIAS_SHIFT, CONV0_OUT_SHIFT, 28 },
    // Conv1: 14x14x32 -> 14x14x32 (3x3 kernel, pad=1, stride=1)
    { conv1_wt, conv1_bias, 14, 32, 32, 3, 1, 1, CONV1_BIAS_SHIFT, CONV1_OUT_SHIFT, 14 },
    // Conv2: 7x7x32 -> 7x7x64     (3x3 kernel, pad=1, stride=1)
    { conv2_wt, conv2_bias,  7, 32, 64, 3, 1, 1, CONV2_BIAS_SHIFT, CONV2_OUT_SHIFT,  7 },
    // Conv3: 7x7x64 -> 7x7x64     (3x3 kernel, pad=1, stride=1)
    { conv3_wt, conv3_bias,  7, 64, 64, 3, 1, 1, CONV3_BIAS_SHIFT, CONV3_OUT_SHIFT,  7 },
};

static const PoolConfig_t mnist_pool_layers[] = {
    // Pool0: 28x28x32 -> 14x14x32 (2x2 window, stride=2)
    { 28, 32, 2, 0, 2, 14 },
    // Pool1: 14x14x32 -> 7x7x32   (2x2 window, stride=2)
    { 14, 32, 2, 0, 2,  7 },
    // Pool2: 7x7x64 -> 4x4x64     (4x4 window, stride=1 - final pool that
    //                              also does the flatten-friendly resize)
    {  7, 64, 4, 0, 1,  4 },
};

static const FCConfig_t mnist_fc_layers[] = {
    // FC1: 1024 (=4*4*64 flattened) -> 256
    { fc1_wt, fc1_bias, 1024, 256, FC1_BIAS_SHIFT, FC1_OUT_SHIFT },
    // FCo: 256 -> 10 (one score per digit class)
    { fco_wt, fco_bias,  256,  10, FCO_BIAS_SHIFT, FCO_OUT_SHIFT },
};

// mnist_network[] - the full ordered layer sequence for this model.
//
// Column meanings (matches Layer_t fields in model_config.h):
//   type          config pointer            in  out  checkpoint  output_size
// Buffer indices (in/out columns) implement the ping-pong pattern by
// hand: each conv/pool/fc alternates 0<->1, and each RELU is in-place
// (same index for in and out) on whichever buffer its preceding layer
// just wrote to.

// checkpoint=1 on every CONV/POOL/FC layer (expensive, worth persisting
// to MRAM); checkpoint=0 on every RELU/SOFTMAX (cheap, re-executed on
// recovery instead of restored - see main.c cnn_inference() comments).

static const Layer_t mnist_network[] = {
    // type            config                    in out cp  out_size
    { LAYER_CONV,    &mnist_conv_layers[0],   0,  0,  1, 28*28*32 },  // Conv0: input -> act0
    { LAYER_RELU,    NULL,                    0,  0,  0, 28*28*32 },  // ReLU on act0 (in-place)
    { LAYER_POOL,    &mnist_pool_layers[0],   0,  1,  1, 14*14*32 },  // Pool0: act0 -> act1
    { LAYER_CONV,    &mnist_conv_layers[1],   1,  0,  1, 14*14*32 },  // Conv1: act1 -> act0
    { LAYER_RELU,    NULL,                    0,  0,  0, 14*14*32 },  // ReLU on act0 (in-place)
    { LAYER_POOL,    &mnist_pool_layers[1],   0,  1,  1,  7*7*32  },  // Pool1: act0 -> act1
    { LAYER_CONV,    &mnist_conv_layers[2],   1,  0,  1,  7*7*64  },  // Conv2: act1 -> act0
    { LAYER_RELU,    NULL,                    0,  0,  0,  7*7*64  },  // ReLU on act0 (in-place)
    { LAYER_CONV,    &mnist_conv_layers[3],   0,  1,  1,  7*7*64  },  // Conv3: act0 -> act1
    { LAYER_RELU,    NULL,                    1,  1,  0,  7*7*64  },  // ReLU on act1 (in-place)
    { LAYER_POOL,    &mnist_pool_layers[2],   1,  0,  1,  4*4*64  },  // Pool2: act1 -> act0
    { LAYER_FC,      &mnist_fc_layers[0],     0,  1,  1,  256     },  // FC1:   act0 -> act1
    { LAYER_RELU,    NULL,                    1,  1,  0,  256     },  // ReLU on act1 (in-place)
    { LAYER_FC,      &mnist_fc_layers[1],     1,  0,  1,  10      },  // FCo:   act1 -> act0
    { LAYER_SOFTMAX, NULL,                    0,  0,  0,  10      },  // Softmax on act0 (in-place, final)
};

// This is the object main.c points its `const ModelConfig_t *model` at.
static const ModelConfig_t mnist_model = {
    .name         = "MNIST_CNN",
    .layers       = mnist_network,
    .num_layers   = sizeof(mnist_network) / sizeof(mnist_network[0]),
    .num_classes  = 10,
    .input_w      = 28,
    .input_h      = 28,
    .input_ch     = 1,
    .max_act_size = 28*28*32,
};

#endif // MNIST_MODEL_H