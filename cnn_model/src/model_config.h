#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include "arm_nnfunctions.h"
#include "weights.h"

//*****************************************************************************
// Generic constants (hardware limits, not model specific)
//*****************************************************************************
#define MAX_ACTIVATION_SIZE  (28*28*64)   // safe upper bound for small CNNs
#define MAX_KERNEL_SIZE      (3)
#define MAX_INPUT_CH         (64)
#define SCRATCH_Q15_SIZE \
    (2 * MAX_KERNEL_SIZE * MAX_KERNEL_SIZE * MAX_INPUT_CH)

//*****************************************************************************
// MRAM addresses (Stage 3 will make these automatic)
//*****************************************************************************
#define MRAM_BASE_ADDR   0x0006F000
#define MRAM_CHECKPOINT  ((uint32_t *)(MRAM_BASE_ADDR + 0x0000))
#define MRAM_BUFFERS_START  (MRAM_BASE_ADDR + 0x1000)

//*****************************************************************************
// Layer types
//*****************************************************************************
typedef enum {
    LAYER_CONV,
    LAYER_POOL,
    LAYER_FC,
    LAYER_RELU,      // in-place
    LAYER_SOFTMAX,   // in-place
    LAYER_FLATTEN    // no-op for HWC (future use)
} LayerType_t;

//*****************************************************************************
// Layer configs
//*****************************************************************************
typedef struct {
    const q7_t *weights;
    const q7_t *bias;
    uint16_t    in_dim;
    uint16_t    in_ch;
    uint16_t    out_ch;
    uint8_t     kernel;
    uint8_t     padding;
    uint8_t     stride;
    uint8_t     bias_shift;
    uint8_t     out_shift;
    uint16_t    out_dim;
} ConvConfig_t;

typedef struct {
    uint16_t    in_dim;
    uint16_t    ch;
    uint8_t     kernel;
    uint8_t     padding;
    uint8_t     stride;
    uint16_t    out_dim;
} PoolConfig_t;

typedef struct {
    const q7_t *weights;
    const q7_t *bias;
    uint16_t    in_size;
    uint16_t    out_size;
    uint8_t     bias_shift;
    uint8_t     out_shift;
} FCConfig_t;

//*****************************************************************************
// Generic layer descriptor
//*****************************************************************************
typedef struct {
    LayerType_t  type;               // CONV / POOL / FC / RELU / SOFTMAX
    const void  *config;             // points to Conv/Pool/FC config (NULL for RELU etc)
    uint8_t      input_buffer;       // 0 = act0, 1 = act1
    uint8_t      output_buffer;      // 0 = act0, 1 = act1
    uint8_t      checkpoint;  // MRAM address (NULL = no checkpoint)
    uint32_t     output_size;        // output size in bytes
} Layer_t;

//*****************************************************************************
// Model descriptor
//*****************************************************************************
typedef struct {
    const char    *name;
    const Layer_t *layers;
    uint16_t       num_layers;
    uint16_t       num_classes;
    uint16_t       input_w;
    uint16_t       input_h;
    uint16_t       input_ch;
    uint32_t       max_act_size;
} ModelConfig_t;

//*****************************************************************************
// MNIST layer configs
//*****************************************************************************
static const ConvConfig_t conv_layers[] = {
    // Conv0: 28x28x1 -> 28x28x32
    { conv0_wt, conv0_bias, 28, 1,  32, 3, 1, 1, CONV0_BIAS_SHIFT, CONV0_OUT_SHIFT, 28 },
    // Conv1: 14x14x32 -> 14x14x32
    { conv1_wt, conv1_bias, 14, 32, 32, 3, 1, 1, CONV1_BIAS_SHIFT, CONV1_OUT_SHIFT, 14 },
    // Conv2: 7x7x32 -> 7x7x64
    { conv2_wt, conv2_bias,  7, 32, 64, 3, 1, 1, CONV2_BIAS_SHIFT, CONV2_OUT_SHIFT,  7 },
    // Conv3: 7x7x64 -> 7x7x64
    { conv3_wt, conv3_bias,  7, 64, 64, 3, 1, 1, CONV3_BIAS_SHIFT, CONV3_OUT_SHIFT,  7 },
};

static const PoolConfig_t pool_layers[] = {
    { 28, 32, 2, 0, 2, 14 },   // Pool0: 28x28x32 -> 14x14x32
    { 14, 32, 2, 0, 2,  7 },   // Pool1: 14x14x32 -> 7x7x32
    {  7, 64, 4, 0, 1,  4 },   // Pool2: 7x7x64  -> 4x4x64
};

static const FCConfig_t fc_layers[] = {
    { fc1_wt, fc1_bias, 1024, 256, FC1_BIAS_SHIFT, FC1_OUT_SHIFT },  // FC1
    { fco_wt, fco_bias,  256,  10, FCO_BIAS_SHIFT, FCO_OUT_SHIFT },  // FCo
};

//*****************************************************************************
// MNIST network sequence (RELU and SOFTMAX explicit!)
//*****************************************************************************
static const Layer_t network[] = {
    // Conv0: input -> act0, ReLU in-place
    { LAYER_CONV,    &conv_layers[0], 0, 0, 1, 28*28*32 },
    { LAYER_RELU,    NULL,            0, 0, 0, 28*28*32 },
    // Pool0: act0 -> act1
    { LAYER_POOL,    &pool_layers[0], 0, 1, 1, 14*14*32 },
    // Conv1: act1 -> act0, ReLU
    { LAYER_CONV,    &conv_layers[1], 1, 0, 1, 14*14*32 },
    { LAYER_RELU,    NULL,            0, 0, 0, 14*14*32 },
    // Pool1: act0 -> act1
    { LAYER_POOL,    &pool_layers[1], 0, 1, 1,  7*7*32  },
    // Conv2: act1 -> act0, ReLU
    { LAYER_CONV,    &conv_layers[2], 1, 0, 1,  7*7*64  },
    { LAYER_RELU,    NULL,            0, 0, 0,  7*7*64  },
    // Conv3: act0 -> act1, ReLU
    { LAYER_CONV,    &conv_layers[3], 0, 1, 1,  7*7*64  },
    { LAYER_RELU,    NULL,            1, 1, 0,  7*7*64  },
    // Pool2: act1 -> act0
    { LAYER_POOL,    &pool_layers[2], 1, 0, 1,  4*4*64  },
    // FC1: act0 -> act1, ReLU
    { LAYER_FC,      &fc_layers[0],   0, 1, 1,  256     },
    { LAYER_RELU,    NULL,            1, 1, 0,  256     },
    // FCo: act1 -> act0
    { LAYER_FC,      &fc_layers[1],   1, 0, 1,  10      },
    // Softmax in-place (final)
    { LAYER_SOFTMAX, NULL,            0, 0, 0,  10      },
};

//*****************************************************************************
// MNIST model instance
//*****************************************************************************
static const ModelConfig_t mnist_model = {
    .name         = "MNIST_CNN",
    .layers       = network,
    .num_layers   = sizeof(network) / sizeof(network[0]),
    .num_classes  = 10,
    .input_w      = 28,
    .input_h      = 28,
    .input_ch     = 1,
    .max_act_size = 28*28*32,
};

#endif // MODEL_CONFIG_H