#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include "arm_nnfunctions.h"
#include "weights.h"


// Layer types
typedef enum {
    LAYER_CONV,
    LAYER_POOL,
    LAYER_FC
} LayerType_t;

// Convolution layer config
typedef struct {
    const q7_t *weights;
    const q7_t *bias;
    uint16_t    in_dim;      // input width/height
    uint16_t    in_ch;       // input channels
    uint16_t    out_ch;      // output channels
    uint8_t     kernel;      // kernel size
    uint8_t     padding;
    uint8_t     stride;
    uint8_t     bias_shift;
    uint8_t     out_shift;
    uint16_t    out_dim;     // output width/height
} ConvConfig_t;

// Pool layer config
typedef struct {
    uint16_t    in_dim;      // input width/height
    uint16_t    ch;          // channels
    uint8_t     kernel;
    uint8_t     padding;
    uint8_t     stride;
    uint16_t    out_dim;     // output width/height
} PoolConfig_t;

// FC layer config
typedef struct {
    const q7_t *weights;
    const q7_t *bias;
    uint16_t    in_size;
    uint16_t    out_size;
    uint8_t     bias_shift;
    uint8_t     out_shift;
} FCConfig_t;

// Model config
#define NUM_CONV_LAYERS  4
#define NUM_POOL_LAYERS  3
#define NUM_FC_LAYERS    2
#define NUM_CLASSES      10


// Conv layers
static const ConvConfig_t conv_layers[NUM_CONV_LAYERS] = {
    // Conv0: 28x28x1 -> 28x28x32
    { conv0_wt, conv0_bias, 28, 1,  32, 3, 1, 1, CONV0_BIAS_SHIFT, CONV0_OUT_SHIFT, 28 },
    // Conv1: 14x14x32 -> 14x14x32
    { conv1_wt, conv1_bias, 14, 32, 32, 3, 1, 1, CONV1_BIAS_SHIFT, CONV1_OUT_SHIFT, 14 },
    // Conv2: 7x7x32 -> 7x7x64
    { conv2_wt, conv2_bias,  7, 32, 64, 3, 1, 1, CONV2_BIAS_SHIFT, CONV2_OUT_SHIFT,  7 },
    // Conv3: 7x7x64 -> 7x7x64
    { conv3_wt, conv3_bias,  7, 64, 64, 3, 1, 1, CONV3_BIAS_SHIFT, CONV3_OUT_SHIFT,  7 },
}; 


// Pool layers
static const PoolConfig_t pool_layers[NUM_POOL_LAYERS] = {
    // Pool0: 28x28x32 -> 14x14x32
    { 28, 32, 2, 0, 2, 14 },
    // Pool1: 14x14x32 -> 7x7x32
    { 14, 32, 2, 0, 2,  7 },
    // Pool2: 7x7x64 -> 4x4x64
    {  7, 64, 4, 0, 1,  4 },
};


// FC layers
static const FCConfig_t fc_layers[NUM_FC_LAYERS] = {
    // FC1: 1024 -> 256
    { fc1_wt, fc1_bias, 1024, 256, FC1_BIAS_SHIFT, FC1_OUT_SHIFT },
    // FCo: 256 -> 10
    { fco_wt, fco_bias,  256,  10, FCO_BIAS_SHIFT, FCO_OUT_SHIFT },
};
#endif // MODEL_CONFIG_H