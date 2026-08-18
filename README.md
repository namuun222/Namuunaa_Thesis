# Namuunaa_Thesis

Intermittent computing on the Apollo4 Blue Lite: a checkpointed CNN inference
engine that survives power failure by persisting per-layer state to MRAM and
resuming from the last completed layer on reboot.

## Environment
| Tool | Details |
|------|---------|
| Board | Ambiq Apollo4 Blue Lite (AMA4B2KP-KBR) |
| IDE | Keil uVision5 |
| Debugger | SEGGER J-Link |
| SDK | AmbiqSuite R4.5.0 |
| Terminal | PuTTY (COM3, 115200 baud) |
| Library | CMSIS-DSP (arm_cortexM4lf_math.lib), CMSIS-NN |

---

## Practice Projects

### 1. Hello World UART
First contact with the board. Verifies UART communication and board initialization.

**What It Does:**
- Initializes cache and low power mode
- Sets up UART at 115200-8-N-1
- Hooks printf to UART via uart_print()
- Prints device info and security info
- Enters deep sleep

**Key Functions:**
| Function | Purpose |
|----------|---------|
| am_hal_cachectrl_config() | Cache setup |
| am_bsp_low_power_init() | Board low power config |
| am_hal_uart_initialize() | UART initialization |
| am_hal_uart_configure() | UART settings |
| am_hal_gpio_pinconfig() | TX/RX pin setup |
| am_util_stdio_printf_init() | Hook printf to UART |
| am_hal_sysctrl_sleep() | Deep sleep |

---

### 2. NVM Read/Write
Demonstrates non-volatile memory operation.
Core mechanism of intermittent computing — proves data survives power loss.

**What It Does:**
- Writes `0xDEADBEEF` to MRAM at `0x00080000`
- Reads back from same address
- Verifies data matches
- Prints result via UART

**Key Functions:**
| Function | Purpose |
|----------|---------|
| am_hal_mram_main_words_program() | Write to MRAM |
| *(uint32_t *)address | Read from MRAM (direct access) |

**Memory Map:**
```
0x00000000  → Start of MRAM (code lives here)
0x00080000  → Data storage (safe distance from code)
0x001FFFFF  → End of MRAM (2MB total)
```

**Key Concepts:**
- MRAM = non-volatile, survives power loss
- SRAM = volatile, lost on power loss
- Reading MRAM = direct memory access (no special function needed)
- Writing MRAM = requires HAL function + security key
- Security key = `AM_HAL_MRAM_PROGRAM_KEY` (0x12344321)
- Address alignment = last hex digit must be 0, 4, 8, or C
- Little endian byte ordering

---

### 3. 2D Convolution (CMSIS-DSP)
Implements CNN convolution layer on embedded device.
Reads inputs from MRAM, computes convolution, writes output back to MRAM.

**CMSIS-DSP Setup:**
```
Step 1: Include Path
Project → Options for Target → C/C++ → Include Paths
Add: AmbiqSuite_R4.5.0\CMSIS\ARM\Include

Step 2: Add Library
lib folder → Add Existing File
Add: AmbiqSuite_R4.5.0\CMSIS\ARM\Lib\ARM\arm_cortexM4lf_math.lib

Step 3: Add Define
C/C++ → Define: ARM_MATH_CM4
```

**What It Does:**
1. Stores input (4x4) and kernel (3x3) in MRAM
2. Reads them back from MRAM
3. Performs 2D convolution row by row
4. Writes output (6x6) back to MRAM
5. Prints results via UART

**MRAM Layout:**
```
MRAM (NVM)
├── Input feature map   ← stored here
├── Kernel/Filter       ← stored here
└── Output feature map  ← written here after conv

float32:
0x00080000  → Input  (16 words, 64 bytes)
0x00081000  → Kernel (12 words, 48 bytes)
0x00082000  → Output (36 words, 144 bytes)

q15:
0x00083000  → Input  (8 words, 32 bytes)
0x00084000  → Kernel (8 words, 16 bytes)
0x00085000  → Output (18 words, 72 bytes)
```

**Data Type Comparison:**
| Type | Bits | Bytes | Range | Needs FPU |
|------|------|-------|-------|-----------|
| float32 | 32 | 4 | ±3.4×10^38 | YES |
| q15 | 16 | 2 | -1.0 to 1.0 | NO |
| q7 | 8 | 1 | -1.0 to 1.0 | NO |

**Memory Usage (4x4 input):**
| Type | Memory |
|------|--------|
| float32 | 64 bytes |
| q15 | 32 bytes (50% less) |
| q7 | 16 bytes (75% less) |

**Key DSP Functions:**
| Function | Purpose |
|----------|---------|
| arm_conv_f32() | float32 convolution |
| arm_conv_q15() | q15 fixed point convolution |
| arm_conv_q7() | q7 fixed point convolution |

**Normalization:**
```
float32: values 1-16 → stored directly
q15:     values 1-16 → divide by 16 → multiply by 32768
q7:      values 1-16 → divide by 16 → multiply by 127
```

**Conversion Formulas:**
```
float → q15:  q15_val   = float_val * 32768
q15 → float:  float_val = q15_val   / 32768

float → q7:   q7_val    = float_val * 127
q7 → float:   float_val = q7_val    / 127
```

**Output Size Formula:**
```
Output = Input + Kernel - 1
       = 4 + 3 - 1 = 6
Output matrix = 6x6 = 36 values
```

---

## CNN Model (cnn_model/)

A single Keil project with a shared checkpointed inference engine
(`main.c` + `model_config.h`) and three swappable benchmark models. Switching
models means changing the `#include` for the model/weights/test-input headers
and the `model` pointer passed to `cnn_inference()` — the engine itself does
not change.

| Model | Status | Headers |
|-------|--------|---------|
| MNIST | Working, quantized accuracy 6/10 on-board | `mnist_model.h`, `mnist_weights.h`, `mnist_test_inputs.h` |
| KWS (keyword spotting) | Working, 7-layer DNN | `kws_model.h`, `kws_weights.h`, `kws_test_inputs.h` |
| CIFAR | In progress | `cifar_model.h`, `cifar_weights.h`, `cifar_test_inputs.h` |

**Source:** [MNIST-Keras saved models](https://github.com/kj7kunal/MNIST-Keras/tree/master/saved_models) (`MNIST_keras_w_CNN.h5`)

### MNIST Architecture

```
Input:  28x28x1 (grayscale image)
Conv0:  3x3, 32 filters, pad=1, stride=1  → 28x28x32
BN0 + ReLU
Conv1:  3x3, 32 filters, pad=1, stride=1  → 28x28x32
BN1 + ReLU
MaxPool1: 2x2, stride=2                   → 14x14x32
Conv2:  3x3, 64 filters, pad=1, stride=1  → 14x14x64
BN2 + ReLU
Conv3:  3x3, 64 filters, pad=1, stride=1  → 14x14x64
BN3 + ReLU
MaxPool2: 2x2, stride=2                   → 7x7x64
MaxPool3: 4x4, stride=1                   → 4x4x64 = 1024
FC1:    1024 → 256 + ReLU
FCo:    256  → 10
Softmax → predicted digit (0-9)
```

---

## Quantization

Post-training quantization from float32 to **q7 (8-bit signed integer)**:

- BatchNorm layers are **folded** into Conv weights before quantization
- Per-layer activation ranges computed from float32 inference
- Scale factor: `127 / max_activation`
- Shift values computed per layer for CMSIS-NN

**MNIST quantized accuracy on board:** 6/10 test samples correct

---

## Weight Storage

Weights are stored in **MRAM** at explicit fixed addresses using `am_hal_mram_main_program()`. An intermediate SRAM buffer is used to avoid MRAM-to-MRAM copy issues:

```c
// Write one weight array to MRAM
for(i = 0; i < WORDS; i++) {
    buffer[0] = src[i];   // copy to SRAM first
    buffer[1] = 0; buffer[2] = 0; buffer[3] = 0;
    am_hal_mram_main_program(AM_HAL_MRAM_PROGRAM_KEY,
                             buffer, &dst[i], 4);
}
```

---

## Checkpointing (Intermittent Computing Core)

After each layer executes, its output is optionally persisted to MRAM at a
fixed address (`mram_addrs[i]`), and a checkpoint pointer is advanced
(`save_checkpoint(i + 1)`). On reboot, `read_checkpoint()` reports the last
completed layer; layers before it are restored from MRAM instead of
recomputed, and execution resumes from there. Activation-only layers
(ReLU/pooling) have no MRAM address and are simply re-run — cheaper than
saving/restoring them.

Example MRAM checkpoint layout (KWS model, 7 layers):
```
Layer 0: 0x00070000 (144 bytes)
Layer 2: 0x00070090 (144 bytes)
Layer 4: 0x00070120 (144 bytes)
Layer 6: 0x000701B0 (12 bytes)
Total checkpoint MRAM: 448 bytes
```

**MRAM write strategy:** writes are batched into groups of 4 words
(`am_hal_mram_main_program` bulk calls) rather than issued one word at a
time, which reduces per-call fixed overhead. This dropped measured
checkpoint overhead from ~5.6% to ~1.9% of total inference time on the KWS
model (see report for full timing methodology and results).

---

## Include Paths

Add the following to **Project → Options for Target → C/C++ → Include Paths**:

```
C:\Users\Dell\Namuunaa_Thesis\CMSIS_5-5.9.0\CMSIS_5-5.9.0\CMSIS\NN\Include
C:\Users\Dell\Namuunaa_Thesis\CMSIS_5-5.9.0\CMSIS_5-5.9.0\CMSIS\Core\Include
```

## Preprocessor Defines

Add the following to **Project → Options for Target → C/C++ → Define**:

```
AM_PACKAGE_BGA AM_PART_APOLLO4L keil6 ARM_MATH_CM4 ARM_MATH_DSP
```

## CMSIS-NN Source Files

Create a group `cnn_mnist` in the project and add these source files:

```
CMSIS_5-5.9.0\CMSIS\NN\Source\ConvolutionFunctions\
    arm_convolve_HWC_q7_basic.c
    arm_convolve_HWC_q7_fast.c
    arm_nn_mat_mult_kernel_q7_q15.c

CMSIS_5-5.9.0\CMSIS\NN\Source\FullyConnectedFunctions\
    arm_fully_connected_q7.c

CMSIS_5-5.9.0\CMSIS\NN\Source\PoolingFunctions\
    arm_pool_q7_HWC.c

CMSIS_5-5.9.0\CMSIS\NN\Source\ActivationFunctions\
    arm_relu_q7.c

CMSIS_5-5.9.0\CMSIS\NN\Source\SoftmaxFunctions\
    arm_softmax_q7.c
```

---

## Source Files

| File | Description |
|------|-------------|
| `src/main.c` | Main inference driver — UART init, timer setup, checkpointed CNN inference, experiment runner |
| `src/model_config.h` | Shared `ModelConfig_t` / `Layer_t` structures and engine config |
| `src/mnist_model.h`, `src/kws_model.h`, `src/cifar_model.h` | Per-model architecture description |
| `src/mnist_weights.h`, `src/kws_weights.h`, `src/cifar_weights.h` | Quantized q7 weights, biases, and shift values |
| `src/mnist_test_inputs.h`, `src/kws_test_inputs.h`, `src/cifar_test_inputs.h` | Real test samples per model |
| `src/am_resources.c` | Board/peripheral resource definitions |

---

## CMSIS-NN Functions Used

| Function | Purpose |
|----------|---------|
| `arm_convolve_HWC_q7_basic()` | First conv layer (1 input channel) |
| `arm_convolve_HWC_q7_fast()` | Subsequent conv layers (SIMD optimized) |
| `arm_maxpool_q7_HWC()` | MaxPool layers |
| `arm_relu_q7()` | ReLU activation |
| `arm_fully_connected_q7()` | Fully connected layers |
| `arm_softmax_q7()` | Output probabilities |

---

## Timing & Energy Analysis

Per-layer execution, checkpoint-save, and recovery-restore times are measured
using the STIMER free-running counter (6 MHz), with energy estimated from
datasheet current draw. Three experiments per model:

- **A** — continuous power, no checkpointing (baseline cost)
- **B** — checkpointing enabled, no failure (checkpoint overhead)
- **C** — recovery cost as a function of failure point (justifies checkpointing)

Full methodology, per-model results, and the recovery-cost crossover analysis
are in the thesis report.
