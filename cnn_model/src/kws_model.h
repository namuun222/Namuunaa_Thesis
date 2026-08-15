#ifndef KWS_MODEL_H
#define KWS_MODEL_H

#include "model_config.h"
#include "kws_weights.h"

//*****************************************************************************
// Keyword Spotting (KWS) plug-in for the generic runtime.
//
// Model: ARM ML-KWS-for-MCU pretrained DNN ("Hello Edge" paper), trained on
// Google Speech Commands. Input: 1 second of 16 kHz audio converted to
// 25 frames x 10 MFCC coefficients = 250 q7 features (Q5.2 format).
// 12 output classes:
//   0:Silence 1:Unknown 2:yes 3:no 4:up 5:down
//   6:left    7:right   8:on  9:off 10:stop 11:go
//*****************************************************************************

static const FCConfig_t kws_fc_layers[] = {
    // IP1: 250 -> 144
    { kws_ip1_wt, kws_ip1_bias, 250, 144, KWS_IP1_BIAS_SHIFT, KWS_IP1_OUT_SHIFT },
    // IP2: 144 -> 144
    { kws_ip2_wt, kws_ip2_bias, 144, 144, KWS_IP2_BIAS_SHIFT, KWS_IP2_OUT_SHIFT },
    // IP3: 144 -> 144
    { kws_ip3_wt, kws_ip3_bias, 144, 144, KWS_IP3_BIAS_SHIFT, KWS_IP3_OUT_SHIFT },
    // IP4: 144 -> 12 (final logits, no softmax - argmax works on raw scores)
    { kws_ip4_wt, kws_ip4_bias, 144,  12, KWS_IP4_BIAS_SHIFT, KWS_IP4_OUT_SHIFT },
};

static const Layer_t kws_network[] = {
    // type          config               in out cp  out_size
    { LAYER_FC,    &kws_fc_layers[0],   0,  0,  1, 144 },  // IP1: input -> act0
    { LAYER_RELU,  NULL,                0,  0,  0, 144 },  // ReLU (in-place)
    { LAYER_FC,    &kws_fc_layers[1],   0,  1,  1, 144 },  // IP2: act0 -> act1
    { LAYER_RELU,  NULL,                1,  1,  0, 144 },  // ReLU (in-place)
    { LAYER_FC,    &kws_fc_layers[2],   1,  0,  1, 144 },  // IP3: act1 -> act0
    { LAYER_RELU,  NULL,                0,  0,  0, 144 },  // ReLU (in-place)
    { LAYER_FC,    &kws_fc_layers[3],   0,  1,  1,  12 },  // IP4: act0 -> act1
};

static const ModelConfig_t kws_model = {
    .name         = "KWS_DNN_ARM",
    .layers       = kws_network,
    .num_layers   = sizeof(kws_network) / sizeof(kws_network[0]),
    .num_classes  = 12,
    .input_w      = 10,     // MFCC coefficients per frame
    .input_h      = 25,     // frames
    .input_ch     = 1,
    .max_act_size = 250,
};

#endif // KWS_MODEL_H
