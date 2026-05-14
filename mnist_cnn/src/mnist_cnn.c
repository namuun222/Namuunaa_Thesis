#include "am_mcu_apollo.h"
#include "am_bsp.h"
#include "am_util.h"
#include "am_hal_global.h"
#include "arm_nnfunctions.h"
#include "weights.h"
#include "mnist_test_inputs.h"

// MRAM addresses for weights

#define MRAM_CONV0_WT    ((q7_t *)0x00090000)
#define MRAM_CONV0_BIAS  ((q7_t *)0x00090200)

// Weight sizes in words

#define CONV0_WT_WORDS    72    // 3*3*1*32 = 288 bytes / 4
#define CONV0_BIAS_WORDS  8     // 32 bytes / 4

// UART handle
void *phUART;

#define CHECK_ERRORS(x)                                                       \
    if ((x) != AM_HAL_STATUS_SUCCESS)                                         \
    {                                                                         \
        error_handler(x);                                                     \
    }

volatile uint32_t ui32LastError;

// Catch HAL errors

void error_handler(uint32_t ui32ErrorStatus)
{
    ui32LastError = ui32ErrorStatus;
    while (1);
}

uint8_t g_pui8TxBuffer[256];
uint8_t g_pui8RxBuffer[2];

//*****************************************************************************
// UART configuration - 115200-8-N-1
//*****************************************************************************
const am_hal_uart_config_t g_sUartConfig =
{
    .ui32BaudRate    = 115200,
    .eDataBits       = AM_HAL_UART_DATA_BITS_8,
    .eParity         = AM_HAL_UART_PARITY_NONE,
    .eStopBits       = AM_HAL_UART_ONE_STOP_BIT,
    .eFlowControl    = AM_HAL_UART_FLOW_CTRL_NONE,
    .eTXFifoLevel    = AM_HAL_UART_FIFO_LEVEL_16,
    .eRXFifoLevel    = AM_HAL_UART_FIFO_LEVEL_16,
};

//*****************************************************************************
// UART interrupt handler
//*****************************************************************************
#if AM_BSP_UART_PRINT_INST == 0
void am_uart_isr(void)
#elif AM_BSP_UART_PRINT_INST == 1
void am_uart1_isr(void)
#elif AM_BSP_UART_PRINT_INST == 2
void am_uart2_isr(void)
#elif AM_BSP_UART_PRINT_INST == 3
void am_uart3_isr(void)
#endif
{
    uint32_t ui32Status;
    am_hal_uart_interrupt_status_get(phUART, &ui32Status, true);
    am_hal_uart_interrupt_clear(phUART, ui32Status);
    am_hal_uart_interrupt_service(phUART, ui32Status);
}

//*****************************************************************************
// UART print string
//*****************************************************************************
void uart_print(char *pcStr)
{
    uint32_t ui32StrLen = 0;
    uint32_t ui32BytesWritten = 0;
    while (pcStr[ui32StrLen] != 0) ui32StrLen++;

    const am_hal_uart_transfer_t sUartWrite =
    {
        .eType                 = AM_HAL_UART_BLOCKING_WRITE,
        .pui8Data              = (uint8_t *)pcStr,
        .ui32NumBytes          = ui32StrLen,
        .pui32BytesTransferred = &ui32BytesWritten,
        .ui32TimeoutMs         = 100,
        .pfnCallback           = NULL,
        .pvContext             = NULL,
        .ui32ErrorStatus       = 0
    };
    CHECK_ERRORS(am_hal_uart_transfer(phUART, &sUartWrite));
    if (ui32BytesWritten != ui32StrLen) while(1);
}

// CNN feature map buffers (SRAM - temporary per layer)
// Weights stored in MRAM as const arrays (weights.h)

static q7_t buf1[28*28*32];   // Conv0 output: 28x28x32
static q7_t buf2[14*14*32];   // Pool0 output: 14x14x32
static q7_t buf3[14*14*32];   // Conv1 output: 14x14x32
static q7_t buf4[7*7*32];     // Pool1 output: 7x7x32
static q7_t buf5[7*7*64];     // Conv2 output: 7x7x64
static q7_t buf6[7*7*64];     // Conv3 output: 7x7x64
static q7_t buf7[4*4*64];     // Pool2 output: 4x4x64 = 1024
static q7_t buf8[256];        // FC1  output: 256
static q7_t output[10];       // FCo  output: 10 class scores

// Scratch buffer for CMSIS-NN convolution intermediate calculations
static q15_t scratch[2*9*64*2];

// Store weights to MRAM using SRAM intermediate buffer

void store_weights_to_mram(void)
{
    uint32_t buffer[4];
    uint32_t *src;
    uint32_t *dst;
    int i;

    //am_util_stdio_printf("=== Storing weights to MRAM ===\n");
    //am_hal_uart_tx_flush(phUART);

    // Write conv0_wt word by word via SRAM buffer
    am_util_stdio_printf("Writing conv0_wt...\n");
    am_hal_uart_tx_flush(phUART);

    src = (uint32_t *)conv0_wt;
    dst = (uint32_t *)MRAM_CONV0_WT;
    for(i = 0; i < CONV0_WT_WORDS; i++)
    {
        buffer[0] = src[i];
        buffer[1] = 0;
        buffer[2] = 0;
        buffer[3] = 0;
        am_hal_mram_main_program(AM_HAL_MRAM_PROGRAM_KEY,
                                 buffer,
                                 &dst[i],
                                 4);
    }

    // Verify - compare first few values
    am_util_stdio_printf("Verify conv0_wt:\n");
    am_util_stdio_printf("  orig[0]=%d  mram[0]=%d\n",
        conv0_wt[0], ((q7_t*)MRAM_CONV0_WT)[0]);
    am_util_stdio_printf("  orig[1]=%d  mram[1]=%d\n",
        conv0_wt[1], ((q7_t*)MRAM_CONV0_WT)[1]);
    am_util_stdio_printf("  orig[2]=%d  mram[2]=%d\n",
        conv0_wt[2], ((q7_t*)MRAM_CONV0_WT)[2]);
    am_hal_uart_tx_flush(phUART);
}

int main(void)
{
    //
    // Set the default cache configuration
    //
    am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
    am_hal_cachectrl_enable();

    //
    // Configure the board for low power operation
    //
    am_bsp_low_power_init();

    //
    // Initialize the printf interface for UART output
    //
    CHECK_ERRORS(am_hal_uart_initialize(AM_BSP_UART_PRINT_INST, &phUART));
    CHECK_ERRORS(am_hal_uart_power_control(phUART, AM_HAL_SYSCTRL_WAKE, false));
    CHECK_ERRORS(am_hal_uart_configure(phUART, &g_sUartConfig));

    //
    // Enable the UART pins
    //
    am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_TX, g_AM_BSP_GPIO_COM_UART_TX);
    am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_RX, g_AM_BSP_GPIO_COM_UART_RX);

    //
    // Enable interrupts
    //
    NVIC_SetPriority((IRQn_Type)(UART0_IRQn + AM_BSP_UART_PRINT_INST), AM_IRQ_PRIORITY_DEFAULT);
    NVIC_EnableIRQ((IRQn_Type)(UART0_IRQn + AM_BSP_UART_PRINT_INST));
    am_hal_interrupt_master_enable();

    //
    // Set the main print interface to use the UART print function
    //
    am_util_stdio_printf_init(uart_print);

    am_util_stdio_printf("MNIST CNN Inference\n");
    am_util_stdio_printf("===================\n\n");
    am_util_stdio_printf("Model: MNIST_keras_w_CNN\n");
    am_util_stdio_printf("Quantization: q7 (post-training)\n");
    am_util_stdio_printf("Library: ARM CMSIS-NN\n\n");

    //
    // Test MRAM write with conv0_wt only
    //
    store_weights_to_mram();

    // CNN Architecture:
    // Input:  28x28x1
    // Conv0:  3x3, 32 filters, pad=1 -> 28x28x32
    // Pool0:  2x2 MaxPool            -> 14x14x32
    // Conv1:  3x3, 32 filters, pad=1 -> 14x14x32
    // Pool1:  2x2 MaxPool            -> 7x7x32
    // Conv2:  3x3, 64 filters, pad=1 -> 7x7x64
    // Conv3:  3x3, 64 filters, pad=1 -> 7x7x64
    // Pool2:  4x4 MaxPool, stride=1  -> 4x4x64 = 1024
    // FC1:    1024 -> 256
    // FCo:    256  -> 10
    // Output: 10 class scores (digits 0-9)

    //
    // Conv0: 28x28x1 -> 28x28x32
    //
    am_util_stdio_printf("Running Conv0...\n");
    arm_convolve_HWC_q7_basic(
        test_input_d2,           // input image
        28,                      // input width/height
        1,                       // input channels
        conv0_wt,                // weights (in MRAM)
        32,                      // output channels
        3,                       // kernel size
        1,                       // padding
        1,                       // stride
        conv0_bias,              // bias (in MRAM)
        CONV0_BIAS_SHIFT,        // bias shift
        CONV0_OUT_SHIFT,         // output shift
        buf1,                    // output buffer
        28,                      // output width
        (q15_t*)scratch,         // scratch buffer
        NULL
    );
    arm_relu_q7(buf1, 28*28*32);

    //
    // Pool0: 28x28x32 -> 14x14x32, MaxPool 2x2
    //
    am_util_stdio_printf("Running Pool0...\n");
    arm_maxpool_q7_HWC(buf1, 28, 32, 2, 0, 2, 14, NULL, buf2);

    //
    // Conv1: 14x14x32 -> 14x14x32
    //
    am_util_stdio_printf("Running Conv1...\n");
    arm_convolve_HWC_q7_fast(
        buf2, 14, 32,
        conv1_wt, 32, 3, 1, 1,
        conv1_bias, CONV1_BIAS_SHIFT, CONV1_OUT_SHIFT,
        buf3, 14, (q15_t*)scratch, NULL
    );
    arm_relu_q7(buf3, 14*14*32);

    //
    // Pool1: 14x14x32 -> 7x7x32, MaxPool 2x2
    //
    am_util_stdio_printf("Running Pool1...\n");
    arm_maxpool_q7_HWC(buf3, 14, 32, 2, 0, 2, 7, NULL, buf4);

    //
    // Conv2: 7x7x32 -> 7x7x64
    //
    am_util_stdio_printf("Running Conv2...\n");
    arm_convolve_HWC_q7_fast(
        buf4, 7, 32,
        conv2_wt, 64, 3, 1, 1,
        conv2_bias, CONV2_BIAS_SHIFT, CONV2_OUT_SHIFT,
        buf5, 7, (q15_t*)scratch, NULL
    );
    arm_relu_q7(buf5, 7*7*64);

    //
    // Conv3: 7x7x64 -> 7x7x64
    //
    am_util_stdio_printf("Running Conv3...\n");
    arm_convolve_HWC_q7_fast(
        buf5, 7, 64,
        conv3_wt, 64, 3, 1, 1,
        conv3_bias, CONV3_BIAS_SHIFT, CONV3_OUT_SHIFT,
        buf6, 7, (q15_t*)scratch, NULL
    );
    arm_relu_q7(buf6, 7*7*64);

    //
    // Pool2: 7x7x64 -> 4x4x64, MaxPool 4x4 stride=1
    //
    am_util_stdio_printf("Running Pool2...\n");
    arm_maxpool_q7_HWC(buf6, 7, 64, 4, 0, 1, 4, NULL, buf7);

    //
    // FC1: 1024 -> 256
    //
    am_util_stdio_printf("Running FC1...\n");
    arm_fully_connected_q7(
        buf7, fc1_wt, 1024, 256,
        FC1_BIAS_SHIFT, FC1_OUT_SHIFT,
        fc1_bias, buf8, (q15_t*)scratch
    );
    arm_relu_q7(buf8, 256);

    //
    // FCo: 256 -> 10
    //
    am_util_stdio_printf("Running FCo...\n");
    arm_fully_connected_q7(
        buf8, fco_wt, 256, 10,
        FCO_BIAS_SHIFT, FCO_OUT_SHIFT,
        fco_bias, output, (q15_t*)scratch
    );

    //
    // Apply softmax to get probabilities
    //
    arm_softmax_q7(output, 10, output);

    //
    // Find predicted digit (highest score)
    //
    int predicted = 0;
    q7_t max_val = output[0];
    for (int i = 1; i < 10; i++)
        if (output[i] > max_val) { max_val = output[i]; predicted = i; }

    //
    // Print final results
    //
    am_util_stdio_printf("\nSoftmax Scores:\n");
    for (int i = 0; i < 10; i++)
        am_util_stdio_printf("  Digit %d: %d\n", i, output[i]);

    am_util_stdio_printf("\nPredicted digit: %d\n", predicted);
    am_util_stdio_printf("\nDone!\n");

    //
    // Flush UART and power down
    //
    am_hal_uart_tx_flush(phUART);
    CHECK_ERRORS(am_hal_uart_power_control(phUART, AM_HAL_SYSCTRL_DEEPSLEEP, false));

    //
    // Loop forever while sleeping
    //
    while (1)
        am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
}