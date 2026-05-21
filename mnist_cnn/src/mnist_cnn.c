#include "am_mcu_apollo.h"
#include "am_bsp.h"
#include "am_util.h"
#include "am_hal_global.h"
#include "arm_nnfunctions.h"
#include "weights.h"
#include "mnist_test_inputs.h"
#include "model_config.h"

//*****************************************************************************
// UART handle
//*****************************************************************************
void *phUART;

#define CHECK_ERRORS(x)                                                       \
    if ((x) != AM_HAL_STATUS_SUCCESS)                                         \
    {                                                                         \
        error_handler(x);                                                     \
    }

volatile uint32_t ui32LastError;

void error_handler(uint32_t ui32ErrorStatus)
{
    ui32LastError = ui32ErrorStatus;
    while (1);
}

uint8_t g_pui8TxBuffer[256];
uint8_t g_pui8RxBuffer[2];

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

//*****************************************************************************
// Feature map buffers (SRAM)
//*****************************************************************************
static q7_t buf1[28*28*32];   // Conv0 output
static q7_t buf2[14*14*32];   // Pool0 output
static q7_t buf3[14*14*32];   // Conv1 output
static q7_t buf4[7*7*32];     // Pool1 output
static q7_t buf5[7*7*64];     // Conv2 output
static q7_t buf6[7*7*64];     // Conv3 output
static q7_t buf7[4*4*64];     // Pool2 output
static q7_t buf8[256];        // FC1 output
static q7_t output[10];       // FCo output

// Scratch buffer for CMSIS-NN
static q15_t scratch[2*9*64*2];

// Buffer pointers for generic inference
static q7_t *bufs[] = { buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8 };

//*****************************************************************************
// Generic conv layer runner
//*****************************************************************************
void run_conv(const ConvConfig_t *cfg, q7_t *in, q7_t *out)
{
    if(cfg->in_ch == 1)
    {
        // Use basic version for single channel input
        arm_convolve_HWC_q7_basic(
            in, cfg->in_dim, cfg->in_ch,
            cfg->weights, cfg->out_ch,
            cfg->kernel, cfg->padding, cfg->stride,
            cfg->bias, cfg->bias_shift, cfg->out_shift,
            out, cfg->out_dim,
            (q15_t*)scratch, NULL
        );
    }
    else
    {
        // Use fast version for multi-channel
        arm_convolve_HWC_q7_fast(
            in, cfg->in_dim, cfg->in_ch,
            cfg->weights, cfg->out_ch,
            cfg->kernel, cfg->padding, cfg->stride,
            cfg->bias, cfg->bias_shift, cfg->out_shift,
            out, cfg->out_dim,
            (q15_t*)scratch, NULL
        );
    }
    // ReLU
    arm_relu_q7(out, cfg->out_dim * cfg->out_dim * cfg->out_ch);
}

//*****************************************************************************
// Generic pool layer runner
//*****************************************************************************
void run_pool(const PoolConfig_t *cfg, q7_t *in, q7_t *out)
{
    arm_maxpool_q7_HWC(
        in, cfg->in_dim, cfg->ch,
        cfg->kernel, cfg->padding, cfg->stride,
        cfg->out_dim, NULL, out
    );
}

//*****************************************************************************
// Generic FC layer runner
//*****************************************************************************
void run_fc(const FCConfig_t *cfg, q7_t *in, q7_t *out, int relu)
{
    arm_fully_connected_q7(
        in, cfg->weights,
        cfg->in_size, cfg->out_size,
        cfg->bias_shift, cfg->out_shift,
        cfg->bias, out,
        (q15_t*)scratch
    );
    if(relu)
        arm_relu_q7(out, cfg->out_size);
}

//*****************************************************************************
// Generic CNN inference
//*****************************************************************************

int cnn_inference(const q7_t *input)
{
    // Conv0 ? Pool0
    run_conv(&conv_layers[0], (q7_t*)input, bufs[0]);
    run_pool(&pool_layers[0], bufs[0], bufs[1]);

    // Conv1 ? Pool1
    run_conv(&conv_layers[1], bufs[1], bufs[2]);
    run_pool(&pool_layers[1], bufs[2], bufs[3]);

    // Conv2
    run_conv(&conv_layers[2], bufs[3], bufs[4]);

    // Conv3 ? Pool2
    run_conv(&conv_layers[3], bufs[4], bufs[5]);
    run_pool(&pool_layers[2], bufs[5], bufs[6]);

    // FC1 (with ReLU)
    run_fc(&fc_layers[0], bufs[6], bufs[7], 1);

    // FCo (no ReLU)
    run_fc(&fc_layers[1], bufs[7], output, 0);

    // Softmax
    arm_softmax_q7(output, NUM_CLASSES, output);

    // Find predicted class
    int predicted = 0;
    q7_t max_val = output[0];
    for(int i = 1; i < NUM_CLASSES; i++)
    {
        if(output[i] > max_val)
        {
            max_val = output[i];
            predicted = i;
        }
    }
    return predicted;
}

int main(void)
{
    // Cache and board init
    am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
    am_hal_cachectrl_enable();
    am_bsp_low_power_init();

    // UART init
    CHECK_ERRORS(am_hal_uart_initialize(AM_BSP_UART_PRINT_INST, &phUART));
    CHECK_ERRORS(am_hal_uart_power_control(phUART, AM_HAL_SYSCTRL_WAKE, false));
    CHECK_ERRORS(am_hal_uart_configure(phUART, &g_sUartConfig));
    am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_TX, g_AM_BSP_GPIO_COM_UART_TX);
    am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_RX, g_AM_BSP_GPIO_COM_UART_RX);
    NVIC_SetPriority((IRQn_Type)(UART0_IRQn + AM_BSP_UART_PRINT_INST), AM_IRQ_PRIORITY_DEFAULT);
    NVIC_EnableIRQ((IRQn_Type)(UART0_IRQn + AM_BSP_UART_PRINT_INST));
    am_hal_interrupt_master_enable();
    am_util_stdio_printf_init(uart_print);


    // Run inference
    int predicted = cnn_inference(test_input_d7);

    // Print scores
    am_util_stdio_printf("\nSoftmax Scores:\n");
    for(int i = 0; i < NUM_CLASSES; i++)
        am_util_stdio_printf("  Digit %d: %d\n", i, output[i]);

    am_util_stdio_printf("\nPredicted digit: %d\n", predicted);
    am_util_stdio_printf("\nDone!\n");

    am_hal_uart_tx_flush(phUART);
    CHECK_ERRORS(am_hal_uart_power_control(phUART, AM_HAL_SYSCTRL_DEEPSLEEP, false));

    while(1)
        am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
}