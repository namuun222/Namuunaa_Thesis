#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include "arm_nnfunctions.h"

#define MAX_ACTIVATION_SIZE  (28*28*64)

// The scratch buffer is used internally
// by convolution and must be large enough for the
// largest kernel * largest channel count used by any conv layer.
#define MAX_KERNEL_SIZE      (3)
#define MAX_INPUT_CH         (64)
#define SCRATCH_Q15_SIZE \
    (2 * MAX_KERNEL_SIZE * MAX_KERNEL_SIZE * MAX_INPUT_CH)

// MRAM_CHECKPOINT is a single 4-byte counter recording which layer last
// completed.
//
// MRAM_BUFFERS_START is where the actual per-layer checkpoint DATA
// begins. Individual layer addresses within this region are NOT defined
// here - they are computed automatically at runtime by
// compute_mram_layout() in main.c, based on whatever model is currently selected.
#define MRAM_BASE_ADDR      0x0006F000
#define MRAM_CHECKPOINT     ((uint32_t *)(MRAM_BASE_ADDR + 0x0000))
#define MRAM_BUFFERS_START  (MRAM_BASE_ADDR + 0x1000)


typedef enum {
    LAYER_CONV,
    LAYER_POOL,
    LAYER_FC,
    LAYER_RELU,      
    LAYER_SOFTMAX,   
    LAYER_FLATTEN   
} LayerType_t;

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

typedef struct {
    LayerType_t  type;
    const void  *config;
    uint8_t      input_buffer;
    uint8_t      output_buffer;
    uint8_t      checkpoint;
    uint32_t     output_size;
} Layer_t;

typedef struct {
    const char    *name;
    const Layer_t *layers;	//the ordered array of Layer_t entries
    uint16_t       num_layers;
    uint16_t       num_classes;
    uint16_t       input_w;
    uint16_t       input_h;
    uint16_t       input_ch;
    uint32_t       max_act_size;
} ModelConfig_t;

#endif // MODEL_CONFIG_H