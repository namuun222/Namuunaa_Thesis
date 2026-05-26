#include "am_mcu_apollo.h"
#include "am_bsp.h"
#include "am_util.h"
#include "am_hal_global.h"
#include "arm_nnfunctions.h"
#include "weights.h"
#include "mnist_test_inputs.h"
#include "model_config.h"

// Ping-pong activation buffers (SRAM)

__attribute__((aligned(4)))
static q7_t act0[MAX_ACTIVATION_SIZE];

__attribute__((aligned(4)))
static q7_t act1[MAX_ACTIVATION_SIZE];

__attribute__((aligned(4)))
static q15_t scratch[SCRATCH_Q15_SIZE];

static q7_t output[NUM_CLASSES];

// UART handle
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



// Generic conv layer runner
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

// Generic pool layer runner
void run_pool(const PoolConfig_t *cfg, q7_t *in, q7_t *out)
{
    arm_maxpool_q7_HWC(
        in, cfg->in_dim, cfg->ch,
        cfg->kernel, cfg->padding, cfg->stride,
        cfg->out_dim, NULL, out
    );
}

// Generic FC layer runner
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
// Save buffer to MRAM word by word via SRAM intermediate
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

// Save checkpoint
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

// Read checkpoint
uint32_t read_checkpoint(void)
{
    return *MRAM_CHECKPOINT;
}

void restore_from_mram(q7_t *dst, q7_t *src, uint32_t size_bytes)
{
    for(uint32_t i = 0; i < size_bytes; i++)
        dst[i] = src[i];
}

void verify_mram(q7_t *sram, q7_t *mram, uint32_t size, char *name)
{
    uint32_t errors = 0;
    for(uint32_t i = 0; i < size; i++)
        if(sram[i] != mram[i]) errors++;

    am_util_stdio_printf("%s: %d/%d match\n", name, size-errors, size);
    am_util_stdio_printf("  SRAM[0..3]: %d %d %d %d\n",
        sram[0], sram[1], sram[2], sram[3]);
    am_util_stdio_printf("  MRAM[0..3]: %d %d %d %d\n",
        mram[0], mram[1], mram[2], mram[3]);
    am_hal_uart_tx_flush(phUART);
}
int cnn_inference(const q7_t *input)
{
    uint32_t cp = read_checkpoint();
    am_util_stdio_printf("Checkpoint: %d\n", cp);

    // Conv0: input -> act0
    if(cp < 1) {
        run_conv(&conv_layers[0], (q7_t*)input, act0);
        save_to_mram(act0, MRAM_BUF0, buf_sizes[0]);
        save_checkpoint(1);
				verify_mram(act0, MRAM_BUF0, 8, "Conv0");
        am_util_stdio_printf("Conv0 done\n");
    } else {
        restore_from_mram(act0, MRAM_BUF0, buf_sizes[0]);
        am_util_stdio_printf("Conv0 restored\n");
    }

    // Pool0: act0 -> act1
    if(cp < 2) {
        run_pool(&pool_layers[0], act0, act1);
        save_to_mram(act1, MRAM_BUF1, buf_sizes[1]);
        save_checkpoint(2);
        am_util_stdio_printf("Pool0 done\n");
    } else {
        restore_from_mram(act1, MRAM_BUF1, buf_sizes[1]);
        am_util_stdio_printf("Pool0 restored\n");
    }

    // Conv1: act1 -> act0
    if(cp < 3) {
        run_conv(&conv_layers[1], act1, act0);
        save_to_mram(act0, MRAM_BUF2, buf_sizes[2]);
        save_checkpoint(3);
        am_util_stdio_printf("Conv1 done\n");
    } else {
        restore_from_mram(act0, MRAM_BUF2, buf_sizes[2]);
        am_util_stdio_printf("Conv1 restored\n");
    }

    // Pool1: act0 -> act1
    if(cp < 4) {
        run_pool(&pool_layers[1], act0, act1);
        save_to_mram(act1, MRAM_BUF3, buf_sizes[3]);
        save_checkpoint(4);
        am_util_stdio_printf("Pool1 done\n");
    } else {
        restore_from_mram(act1, MRAM_BUF3, buf_sizes[3]);
        am_util_stdio_printf("Pool1 restored\n");
    }

    // Conv2: act1 -> act0
    if(cp < 5) {
        run_conv(&conv_layers[2], act1, act0);
        save_to_mram(act0, MRAM_BUF4, buf_sizes[4]);
        save_checkpoint(5);
        am_util_stdio_printf("Conv2 done\n");
    } else {
        restore_from_mram(act0, MRAM_BUF4, buf_sizes[4]);
        am_util_stdio_printf("Conv2 restored\n");
    }

    // Conv3: act0 -> act1
    if(cp < 6) {
        run_conv(&conv_layers[3], act0, act1);
        save_to_mram(act1, MRAM_BUF5, buf_sizes[5]);
        save_checkpoint(6);
        am_util_stdio_printf("Conv3 done\n");
    } else {
        restore_from_mram(act1, MRAM_BUF5, buf_sizes[5]);
        am_util_stdio_printf("Conv3 restored\n");
    }

    // Pool2: act1 -> act0
    if(cp < 7) {
        run_pool(&pool_layers[2], act1, act0);
        save_to_mram(act0, MRAM_BUF6, buf_sizes[6]);
        save_checkpoint(7);
        am_util_stdio_printf("Pool2 done\n");
    } else {
        restore_from_mram(act0, MRAM_BUF6, buf_sizes[6]);
        am_util_stdio_printf("Pool2 restored\n");
    }

    // FC1: act0 -> act1
    if(cp < 8) {
        run_fc(&fc_layers[0], act0, act1, 1);
        save_to_mram(act1, MRAM_BUF7, buf_sizes[7]);
        save_checkpoint(8);
        am_util_stdio_printf("FC1 done\n");
    } else {
        restore_from_mram(act1, MRAM_BUF7, buf_sizes[7]);
        am_util_stdio_printf("FC1 restored\n");
    }

    // FCo: act1 -> output (always runs last)
    run_fc(&fc_layers[1], act1, output, 0);
    save_to_mram(output, MRAM_OUTPUT, buf_sizes[8]);
    save_checkpoint(9);
    am_util_stdio_printf("FCo done\n");

    arm_softmax_q7(output, NUM_CLASSES, output);

    int predicted = 0;
    q7_t max_val = output[0];
    for(int i = 1; i < NUM_CLASSES; i++)
        if(output[i] > max_val) { max_val = output[i]; predicted = i; }

    return predicted;
}

/*int cnn_inference(const q7_t *input)
{
    // Conv0: input -> act0
    run_conv(&conv_layers[0], (q7_t*)input, act0);
    save_to_mram(act0, MRAM_BUF0, buf_sizes[0]);

		save_checkpoint(1);
	
    // Pool0: act0 -> act1
    run_pool(&pool_layers[0], act0, act1);
		save_to_mram(act1, MRAM_BUF1, buf_sizes[1]);
		save_checkpoint(2);
    
    // Conv1: act1 -> act0
    run_conv(&conv_layers[1], act1, act0);
    save_to_mram(act0, MRAM_BUF2, buf_sizes[2]);
		save_checkpoint(3);
    // Pool1: act0 -> act1
    run_pool(&pool_layers[1], act0, act1);
	  save_to_mram(act1, MRAM_BUF3, buf_sizes[3]);
    save_checkpoint(4);
    
    // Conv2: act1 -> act0
    run_conv(&conv_layers[2], act1, act0);
    save_to_mram(act0, MRAM_BUF4, buf_sizes[4]);
    save_checkpoint(5);
    // Conv3: act0 -> act1
    run_conv(&conv_layers[3], act0, act1);
    save_to_mram(act1, MRAM_BUF5, buf_sizes[5]);
    save_checkpoint(6);
    // Pool2: act1 -> act0
    run_pool(&pool_layers[2], act1, act0);
    save_to_mram(act0, MRAM_BUF6, buf_sizes[6]);
    save_checkpoint(7);
    // FC1: act0 -> act1
    run_fc(&fc_layers[0], act0, act1, 1);
		save_to_mram(act1, MRAM_BUF7, buf_sizes[7]);
    save_checkpoint(8);

    
    // FCo: act1 -> output
    run_fc(&fc_layers[1], act1, output, 0);
    save_to_mram(output, MRAM_OUTPUT, buf_sizes[8]);
    save_checkpoint(9);
    
		//Softmax
	  arm_softmax_q7(output, NUM_CLASSES, output);
    
    int predicted = 0;
    q7_t max_val = output[0];
    for(int i = 1; i < NUM_CLASSES; i++)
        if(output[i] > max_val) { max_val = output[i]; predicted = i; }
    
    return predicted;
}
*/
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


	    // Check checkpoint
    uint32_t cp = read_checkpoint();
    if(cp == 0xFFFFFFFF || cp == 9)
    {
        am_util_stdio_printf("Fresh start\n");
        save_checkpoint(0);
    }
    else
        am_util_stdio_printf("Resuming from checkpoint %d\n", cp);
		
    // Run inference
    int predicted = cnn_inference(test_input_d5);

    // Print scores
    am_util_stdio_printf("\nSoftmax Scores:\n");
    for(int i = 0; i < NUM_CLASSES; i++)
        am_util_stdio_printf("  Digit %d: %d\n", i, output[i]);

    am_util_stdio_printf("\nPredicted digit: %d\n", predicted);
    am_util_stdio_printf("\nDone!\n");
		
		    save_checkpoint(0xFFFFFFFF);  // reset for next run
   
	 am_hal_uart_tx_flush(phUART);
    CHECK_ERRORS(am_hal_uart_power_control(phUART, AM_HAL_SYSCTRL_DEEPSLEEP, false));

    while(1)
        am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
}