#include "am_mcu_apollo.h"
#include "am_bsp.h"
#include "am_util.h"
#include "am_hal_global.h"
#include "arm_nnfunctions.h"
#include "model_config.h"       
#include "mnist_model.h"        
#include "mnist_test_inputs.h" 

__attribute__((aligned(4)))
static q7_t act0[MAX_ACTIVATION_SIZE];

__attribute__((aligned(4)))
static q7_t act1[MAX_ACTIVATION_SIZE];

__attribute__((aligned(4)))
static q15_t scratch[SCRATCH_Q15_SIZE];

__attribute__((aligned(4)))
static q7_t output[32];

#define MAX_LAYERS  32
#define ALIGN16(x)  (((x) + 15) & ~15)   // round up to next 16-byte boundary

static q7_t *mram_addrs[MAX_LAYERS];

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


// Writes `size_bytes` of data from an SRAM buffer into MRAM, 16 bytes (4 words) at a time
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

// Copies `size_bytes` from MRAM back into an SRAM buffer. Unlike writing,
// READING from MRAM has no special hardware requirements - it behaves
// like normal memory-mapped flash, so a plain byte-by-byte copy is safe.
// Used during checkpoint recovery to bring a previously-completed layer's
// output back into the ping-pong SRAM buffers.
void restore_from_mram(q7_t *dst, q7_t *src, uint32_t size_bytes)
{
    for(uint32_t i = 0; i < size_bytes; i++)
        dst[i] = src[i];
}

// Writes a single 4-byte counter to a fixed MRAM address, recording
// "layer `layer` has just completed." This is the value read back on
// boot to decide whether to resume mid-network or start fresh. 
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

//*****************************************************************************
// read_checkpoint()
//
// Reads back the last saved checkpoint counter from MRAM. Called once at
// boot to determine whether this is a fresh run (0xFFFFFFFF, MRAM's
// erased/default state, or a value equal to the total layer count meaning
// the previous run finished successfully) or a resume after power loss
// (any value less than the model's layer count).
uint32_t read_checkpoint(void)
{
    return *MRAM_CHECKPOINT;
}

// Small helper that maps a Layer_t's input_buffer/output_buffer index
// (0 or 1, as stored in the model description) to the actual act0/act1 pointer. 
//Keeps the ping-pong buffer selection in one place.
static q7_t *get_sram_buffer(uint8_t id)
{
    return (id == 0) ? act0 : act1;
}

// Thin wrappers around save_to_mram()/restore_from_mram() that look up
// the correct SRAM buffer (via the layer's output_buffer index) and the
// correct MRAM address (via mram_addrs[i], computed by
// compute_mram_layout()) for a given layer index `i`. Kept as separate
// helpers so the main inference loop stays readable.
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

// Walks the model's layer array ONCE at boot and assigns each
// checkpointed layer (layer->checkpoint == 1) the next free, 16-byte
// aligned MRAM address, packing them tightly one after another starting
// at MRAM_BUFFERS_START. Layers with checkpoint == 0 (ReLU, Softmax -
// cheap to just re-run) get mram_addrs[i] = NULL and are skipped.
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
            arm_relu_q7(in, layer->output_size);
            break;

        case LAYER_SOFTMAX:
            arm_softmax_q7(in, layer->output_size, in);
            break;

        case LAYER_FLATTEN:
            break;
    }
}
//   For each layer i:
//     - if the saved checkpoint says layer i was NOT completed yet
//       (cp <= i): execute it normally, then (if it's a checkpointed
//       layer) save its output to MRAM and advance the checkpoint.
//     - if layer i WAS already completed in a previous run (cp > i):
//         - if it has a saved MRAM output, restore it directly

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

        // First layer reads the raw input image; every other layer
        // reads whichever ping-pong buffer the model description says.
        if(i == 0)
            in = (q7_t*)input_image;
        else
            in = get_sram_buffer(layer->input_buffer);

        out = get_sram_buffer(layer->output_buffer);

        if(cp <= i)
        {
            execute_layer(layer, in, out);

            if(mram_addrs[i] != NULL)
                save_layer_output(layer, i);

            save_checkpoint(i + 1);
            am_util_stdio_printf("Layer %d done\n", i);
        }
        else
        {
            if(mram_addrs[i] != NULL)
            {
                restore_layer_output(layer, i);
                am_util_stdio_printf("Layer %d restored\n", i);
            }
            else
            {
                execute_layer(layer, in, out);
                am_util_stdio_printf("Layer %d re-executed\n", i);
            }
        }
    }

    const Layer_t *last = &model->layers[model->num_layers - 1];
    q7_t *final = get_sram_buffer(last->output_buffer);

    int predicted = 0;
    q7_t max_val = final[0];
    for(int i = 1; i < model->num_classes; i++)
        if(final[i] > max_val) { max_val = final[i]; predicted = i; }

    for(int i = 0; i < model->num_classes; i++)
        output[i] = final[i];

    return predicted;
}

int main(void)
{
    am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
    am_hal_cachectrl_enable();
    am_bsp_low_power_init();

    CHECK_ERRORS(am_hal_uart_initialize(AM_BSP_UART_PRINT_INST, &phUART));
    CHECK_ERRORS(am_hal_uart_power_control(phUART, AM_HAL_SYSCTRL_WAKE, false));
    CHECK_ERRORS(am_hal_uart_configure(phUART, &g_sUartConfig));
    am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_TX, g_AM_BSP_GPIO_COM_UART_TX);
    am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_RX, g_AM_BSP_GPIO_COM_UART_RX);
    NVIC_SetPriority((IRQn_Type)(UART0_IRQn + AM_BSP_UART_PRINT_INST), AM_IRQ_PRIORITY_DEFAULT);
    NVIC_EnableIRQ((IRQn_Type)(UART0_IRQn + AM_BSP_UART_PRINT_INST));
    am_hal_interrupt_master_enable();
    am_util_stdio_printf_init(uart_print);

    const ModelConfig_t *model = &mnist_model;
   
		compute_mram_layout(model);
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
    int predicted = cnn_inference(model, test_input_d4);

    am_util_stdio_printf("\nSoftmax Scores:\n");
    for(int i = 0; i < model->num_classes; i++)
        am_util_stdio_printf("  Class %d: %d\n", i, output[i]);

    am_util_stdio_printf("\nPredicted: %d\n", predicted);
    am_util_stdio_printf("\nDone!\n");

    save_checkpoint(0xFFFFFFFF);  // reset for next run

    am_hal_uart_tx_flush(phUART);
    CHECK_ERRORS(am_hal_uart_power_control(phUART, AM_HAL_SYSCTRL_DEEPSLEEP, false));

    while(1)
        am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
}