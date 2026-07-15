#include "am_mcu_apollo.h"
#include "am_bsp.h"
#include "am_util.h"
#include "am_hal_global.h"
#include "arm_nnfunctions.h"
#include "weights.h"
#include "mnist_test_inputs.h"
#include "model_config.h"

//*****************************************************************************
// Ping-pong activation buffers (SRAM)
//*****************************************************************************
__attribute__((aligned(4)))
static q7_t act0[MAX_ACTIVATION_SIZE];

__attribute__((aligned(4)))
static q7_t act1[MAX_ACTIVATION_SIZE];

__attribute__((aligned(4)))
static q15_t scratch[SCRATCH_Q15_SIZE];

__attribute__((aligned(4)))
static q7_t output[32];

//*****************************************************************************
// Automatic MRAM layout (computed at startup)
//*****************************************************************************
#define MAX_LAYERS  32
#define ALIGN16(x)  (((x) + 15) & ~15)

static q7_t *mram_addrs[MAX_LAYERS];

//*****************************************************************************
// UART
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
// Layer runners
//*****************************************************************************
void run_conv(const ConvConfig_t *cfg, q7_t *in, q7_t *out)
{
    if(cfg->in_ch == 1)
    {
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
        arm_convolve_HWC_q7_fast(
            in, cfg->in_dim, cfg->in_ch,
            cfg->weights, cfg->out_ch,
            cfg->kernel, cfg->padding, cfg->stride,
            cfg->bias, cfg->bias_shift, cfg->out_shift,
            out, cfg->out_dim,
            (q15_t*)scratch, NULL
        );
    }
}

void run_pool(const PoolConfig_t *cfg, q7_t *in, q7_t *out)
{
    arm_maxpool_q7_HWC(
        in, cfg->in_dim, cfg->ch,
        cfg->kernel, cfg->padding, cfg->stride,
        cfg->out_dim, NULL, out
    );
}

void run_fc(const FCConfig_t *cfg, q7_t *in, q7_t *out)
{
    arm_fully_connected_q7(
        in, cfg->weights,
        cfg->in_size, cfg->out_size,
        cfg->bias_shift, cfg->out_shift,
        cfg->bias, out,
        (q15_t*)scratch
    );
}

//*****************************************************************************
// MRAM primitives
//*****************************************************************************
void save_to_mram(q7_t *src, q7_t *dst, uint32_t size_bytes)
{
    uint32_t buffer[4];
    uint32_t *src32 = (uint32_t *)src;
    uint32_t *dst32 = (uint32_t *)dst;
    uint32_t words  = (size_bytes + 3) / 4;
    uint32_t i;

    for(i = 0; i < words; i += 4)
    {
        buffer[0] = (i+0 < words) ? src32[i+0] : 0;
        buffer[1] = (i+1 < words) ? src32[i+1] : 0;
        buffer[2] = (i+2 < words) ? src32[i+2] : 0;
        buffer[3] = (i+3 < words) ? src32[i+3] : 0;
        am_hal_mram_main_program(AM_HAL_MRAM_PROGRAM_KEY,
                                 buffer,
                                 &dst32[i],
                                 4);
    }
}

void restore_from_mram(q7_t *dst, q7_t *src, uint32_t size_bytes)
{
    for(uint32_t i = 0; i < size_bytes; i++)
        dst[i] = src[i];
}

void save_checkpoint(uint32_t layer)
{
    uint32_t buffer[4] = {layer, 0, 0, 0};
    am_hal_mram_main_program(
        AM_HAL_MRAM_PROGRAM_KEY,
        buffer,
        MRAM_CHECKPOINT,
        4
    );
}

uint32_t read_checkpoint(void)
{
    return *MRAM_CHECKPOINT;
}

//*****************************************************************************
// SRAM buffer selection
//*****************************************************************************
static q7_t *get_sram_buffer(uint8_t id)
{
    return (id == 0) ? act0 : act1;
}

//*****************************************************************************
// Checkpoint helpers (take layer index for MRAM address lookup)
//*****************************************************************************
static void save_layer_output(const Layer_t *layer, uint32_t i)
{
    q7_t *src = get_sram_buffer(layer->output_buffer);
    save_to_mram(src, mram_addrs[i], layer->output_size);
}

static void restore_layer_output(const Layer_t *layer, uint32_t i)
{
    q7_t *dst = get_sram_buffer(layer->output_buffer);
    restore_from_mram(dst, mram_addrs[i], layer->output_size);
}

//*****************************************************************************
// Automatic MRAM layout - computed once at startup
//*****************************************************************************
void compute_mram_layout(const ModelConfig_t *model)
{
    uint32_t addr = MRAM_BUFFERS_START;

    am_util_stdio_printf("MRAM layout:\n");
    for(uint32_t i = 0; i < model->num_layers; i++)
    {
        if(model->layers[i].checkpoint)
        {
            mram_addrs[i] = (q7_t *)addr;
            am_util_stdio_printf("  Layer %2d: 0x%08X  (%d bytes)\n",
                i, addr, model->layers[i].output_size);
            addr += ALIGN16(model->layers[i].output_size);
        }
        else
        {
            mram_addrs[i] = NULL;
        }
    }
    am_util_stdio_printf("Total checkpoint MRAM: %d bytes\n\n",
        addr - MRAM_BUFFERS_START);
}

//*****************************************************************************
// Generic layer execution
//*****************************************************************************
static void execute_layer(const Layer_t *layer, q7_t *in, q7_t *out)
{
    switch(layer->type)
    {
        case LAYER_CONV:
            run_conv((const ConvConfig_t*)layer->config, in, out);
            break;

        case LAYER_POOL:
            run_pool((const PoolConfig_t*)layer->config, in, out);
            break;

        case LAYER_FC:
            run_fc((const FCConfig_t*)layer->config, in, out);
            break;

        case LAYER_RELU:
            // in-place on input (input_buffer == output_buffer)
            arm_relu_q7(in, layer->output_size);
            break;

        case LAYER_SOFTMAX:
            // in-place
            arm_softmax_q7(in, layer->output_size, in);
            break;

        case LAYER_FLATTEN:
            // no-op for HWC format
            break;
    }
}

//*****************************************************************************
// Generic CNN inference with checkpoint and recovery
//*****************************************************************************
int cnn_inference(const ModelConfig_t *model, const q7_t *input_image)
{
    uint32_t cp = read_checkpoint();
    am_util_stdio_printf("Model: %s (%d layers)\n",
                         model->name, model->num_layers);
    am_util_stdio_printf("Checkpoint: %d\n", cp);

    for(uint32_t i = 0; i < model->num_layers; i++)
    {
        const Layer_t *layer = &model->layers[i];
        q7_t *in, *out;

        // Input: image for first layer, else SRAM buffer
        if(i == 0)
            in = (q7_t*)input_image;
        else
            in = get_sram_buffer(layer->input_buffer);

        out = get_sram_buffer(layer->output_buffer);

        if(cp <= i)
        {
            // Normal execution
            execute_layer(layer, in, out);

            if(mram_addrs[i] != NULL)
                save_layer_output(layer, i);

            save_checkpoint(i + 1);
            am_util_stdio_printf("Layer %d done\n", i);
						
						            if(i == 8)
            {
                am_util_stdio_printf(">>> RESET NOW! (5 sec window)\n");
                am_hal_uart_tx_flush(phUART);
                am_util_delay_ms(5000);
            }
        }
        else
        {
            // Recovery path
            if(mram_addrs[i] != NULL)
            {
                restore_layer_output(layer, i);
                am_util_stdio_printf("Layer %d restored\n", i);
            }
            else
            {
                // RELU/SOFTMAX have no saved state - re-run (cheap!)
                execute_layer(layer, in, out);
                am_util_stdio_printf("Layer %d re-executed\n", i);
            }
        }
    }

    // Final output is in the last layer's output buffer
    const Layer_t *last = &model->layers[model->num_layers - 1];
    q7_t *final = get_sram_buffer(last->output_buffer);

    int predicted = 0;
    q7_t max_val = final[0];
    for(int i = 1; i < model->num_classes; i++)
        if(final[i] > max_val) { max_val = final[i]; predicted = i; }

    // Copy to global output for printing
    for(int i = 0; i < model->num_classes; i++)
        output[i] = final[i];

    return predicted;
}

//*****************************************************************************
// Main
//*****************************************************************************
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

    // Model pointer - change this line to switch models!
    const ModelConfig_t *model = &mnist_model;

    // Compute MRAM layout automatically from model!
    compute_mram_layout(model);

    // Check checkpoint status
    uint32_t cp = read_checkpoint();
    if(cp == 0xFFFFFFFF || cp >= model->num_layers)
    {
        am_util_stdio_printf("Fresh start\n");
        save_checkpoint(0);
    }
    else
    {
        am_util_stdio_printf("Resuming from checkpoint %d\n", cp);
    }

    // Run inference
    int predicted = cnn_inference(model, test_input_d7);

    // Print scores
    am_util_stdio_printf("\nSoftmax Scores:\n");
    for(int i = 0; i < model->num_classes; i++)
        am_util_stdio_printf("  Digit %d: %d\n", i, output[i]);

    am_util_stdio_printf("\nPredicted digit: %d\n", predicted);
    am_util_stdio_printf("\nDone!\n");

    save_checkpoint(0xFFFFFFFF);  // reset for next run

    am_hal_uart_tx_flush(phUART);
    CHECK_ERRORS(am_hal_uart_power_control(phUART, AM_HAL_SYSCTRL_DEEPSLEEP, false));

    while(1)
        am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
}