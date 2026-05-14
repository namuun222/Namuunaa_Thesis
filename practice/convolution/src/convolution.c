#include "am_mcu_apollo.h"
#include "am_bsp.h"
#include "am_util.h"
#include "am_hal_global.h"
#include "arm_math.h"

//*****************************************************************************
//
// UART handle.
//
//*****************************************************************************
void *phUART;

#define CHECK_ERRORS(x)                                                       \
    if ((x) != AM_HAL_STATUS_SUCCESS)                                         \
    {                                                                         \
        error_handler(x);                                                     \
    }

volatile uint32_t ui32LastError;

//*****************************************************************************
//
// Catch HAL errors.
//
//*****************************************************************************
void
error_handler(uint32_t ui32ErrorStatus)
{
    ui32LastError = ui32ErrorStatus;

    while (1);
}

//*****************************************************************************
//
// UART buffers.
//
//*****************************************************************************
uint8_t g_pui8TxBuffer[256];
uint8_t g_pui8RxBuffer[2];

//*****************************************************************************
//
// UART configuration.
//
//*****************************************************************************
const am_hal_uart_config_t g_sUartConfig =
{
    //
    // Standard UART settings: 115200-8-N-1
    //
    .ui32BaudRate = 115200,
    .eDataBits = AM_HAL_UART_DATA_BITS_8,
    .eParity = AM_HAL_UART_PARITY_NONE,
    .eStopBits = AM_HAL_UART_ONE_STOP_BIT,
    .eFlowControl = AM_HAL_UART_FLOW_CTRL_NONE,

    //
    // Set TX and RX FIFOs to interrupt at half-full.
    //
    .eTXFifoLevel = AM_HAL_UART_FIFO_LEVEL_16,
    .eRXFifoLevel = AM_HAL_UART_FIFO_LEVEL_16,
};

//*****************************************************************************
//
// UART interrupt handler.
//
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
    //
    // Service the FIFOs as necessary, and clear the interrupts.
    //
    uint32_t ui32Status;
    am_hal_uart_interrupt_status_get(phUART, &ui32Status, true);
    am_hal_uart_interrupt_clear(phUART, ui32Status);
    am_hal_uart_interrupt_service(phUART, ui32Status);
}

//*****************************************************************************
//
// UART print string
//
//*****************************************************************************

void
uart_print(char *pcStr)
{
    uint32_t ui32StrLen = 0;
    uint32_t ui32BytesWritten = 0;

    //
    // Measure the length of the string.
    //
    while (pcStr[ui32StrLen] != 0)
    {
        ui32StrLen++;
    }

    //
    // Print the string via the UART.
    //
    const am_hal_uart_transfer_t sUartWrite =
    {
        .eType = AM_HAL_UART_BLOCKING_WRITE,
        .pui8Data = (uint8_t *) pcStr,
        .ui32NumBytes = ui32StrLen,
        .pui32BytesTransferred = &ui32BytesWritten,
        .ui32TimeoutMs = 100,
        .pfnCallback = NULL,
        .pvContext = NULL,
        .ui32ErrorStatus = 0
    };

    CHECK_ERRORS(am_hal_uart_transfer(phUART, &sUartWrite));

    if (ui32BytesWritten != ui32StrLen)
    {
        //
        // Couldn't send the whole string!!
        //
        while(1);
    }
}

#define ENABLE_DEBUGGER

//*****************************************************************************
//
// Main
//
//*****************************************************************************
int main(void)
{

    //
    // Set the default cache configuration
    //
    am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
    am_hal_cachectrl_enable();

    //
    // Configure the board for low power operation.
    //
    am_bsp_low_power_init();

    //
    // Initialize the printf interface for UART output.
    //
    CHECK_ERRORS(am_hal_uart_initialize(AM_BSP_UART_PRINT_INST, &phUART));
    CHECK_ERRORS(am_hal_uart_power_control(phUART, AM_HAL_SYSCTRL_WAKE, false));
    CHECK_ERRORS(am_hal_uart_configure(phUART, &g_sUartConfig));

    //
    // Enable the UART pins.
    //
    am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_TX, g_AM_BSP_GPIO_COM_UART_TX);
    am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_RX, g_AM_BSP_GPIO_COM_UART_RX);

    //
    // Enable interrupts.
    //
    NVIC_SetPriority((IRQn_Type)(UART0_IRQn + AM_BSP_UART_PRINT_INST), AM_IRQ_PRIORITY_DEFAULT);
    NVIC_EnableIRQ((IRQn_Type)(UART0_IRQn + AM_BSP_UART_PRINT_INST));
    am_hal_interrupt_master_enable();

    //
    // Set the main print interface to use the UART print function we defined.
    //
    am_util_stdio_printf_init(uart_print);

  //convolution
	am_util_stdio_printf("2D Convolution using CMSIS-DSP\n\n");

// --- Input feature map (4x4) -------------------------------
float32_t pSrc[16] =
{
    1.0f, 2.0f, 3.0f, 4.0f,
    5.0f, 6.0f, 7.0f, 8.0f,
    9.0f, 10.0f, 11.0f, 12.0f,
    13.0f, 14.0f, 15.0f, 16.0f
};

// --- Kernel (3x3) ------------------------------------------
float32_t pKernel[9] =
{
    1.0f, 0.0f, -1.0f,
    1.0f, 0.0f, -1.0f,
    1.0f, 0.0f, -1.0f
};

// --- Output size = (4+3-1) x (4+3-1) = 6x6 ----------------
float32_t pDst[36] = {0};

// --- Temporary buffer for row convolution ------------------
float32_t pRowResult[6] = {0};

// --- MRAM storage addresses --------------------------------
uint32_t *pMRAM_Input  = (uint32_t *)0x00080000;
uint32_t *pMRAM_Kernel = (uint32_t *)0x00081000;
uint32_t *pMRAM_Output = (uint32_t *)0x00082000;

// --- Write input to MRAM -----------------------------------
am_util_stdio_printf("Writing input to MRAM...\n");
am_hal_mram_main_words_program(AM_HAL_MRAM_PROGRAM_KEY,
    (uint32_t *)pSrc, pMRAM_Input, 16);

// --- Write kernel to MRAM ----------------------------------
am_util_stdio_printf("Writing kernel to MRAM...\n");
am_hal_mram_main_words_program(AM_HAL_MRAM_PROGRAM_KEY,
    (uint32_t *)pKernel, pMRAM_Kernel, 12);

// --- Read back input from MRAM -----------------------------	`
am_util_stdio_printf("Reading input from MRAM...\n");
float32_t pSrc_read[16];
for (int i = 0; i < 16; i++)
{
    pSrc_read[i] = *(float32_t *)(pMRAM_Input + i);
}

// --- Read back kernel from MRAM ----------------------------
am_util_stdio_printf("Reading kernel from MRAM...\n");
float32_t pKernel_read[9];
for (int i = 0; i < 9; i++)
{
    pKernel_read[i] = *(float32_t *)(pMRAM_Kernel + i);
}

// --- 2D Convolution row by row using CMSIS-DSP -------------
am_util_stdio_printf("Performing 2D Convolution...\n\n");
for (int row = 0; row < 4; row++)
{
    arm_conv_f32(&pSrc_read[row * 4], 4,
                  pKernel_read, 3,
                  &pDst[row * 6]);
}

// --- Write output to MRAM ----------------------------------
am_util_stdio_printf("Writing output to MRAM...\n\n");
am_hal_mram_main_words_program(AM_HAL_MRAM_PROGRAM_KEY,
    (uint32_t *)pDst, pMRAM_Output, 36);

// --- Print output ------------------------------------------
am_util_stdio_printf("Convolution Output (6x6):\n");
for (int i = 0; i < 6; i++)
{
    for (int j = 0; j < 6; j++)
    {
        am_util_stdio_printf("%6.1f ", pDst[i * 6 + j]);
    }
    am_util_stdio_printf("\n");
}
am_util_stdio_printf("\nDone!\n");

// -----------------------------------------------------------
// Q15 CONVOLUTION
// -----------------------------------------------------------

am_util_stdio_printf("\n\n--- Q15 Convolution ---\n\n");

// --- Input (4x4) normalized to q15 ------------------------
q15_t pSrc_q15[16] =
{
    (q15_t)(0.0625f * 32768),
    (q15_t)(0.125f  * 32768),
    (q15_t)(0.1875f * 32768),
    (q15_t)(0.25f   * 32768),
    (q15_t)(0.3125f * 32768),
    (q15_t)(0.375f  * 32768),
    (q15_t)(0.4375f * 32768),
    (q15_t)(0.5f    * 32768),
    (q15_t)(0.5625f * 32768),
    (q15_t)(0.625f  * 32768),
    (q15_t)(0.6875f * 32768),
    (q15_t)(0.75f   * 32768),
    (q15_t)(0.8125f * 32768),
    (q15_t)(0.875f  * 32768),
    (q15_t)(0.9375f * 32768),
    (q15_t)(1.0f    * 32767)
};

// --- Kernel (3x3) ------------------------------------------
q15_t pKernel_q15[12] =
{
    (q15_t)(1.0f  * 32767),
    (q15_t)(0.0f  * 32767),
    (q15_t)(-1.0f * 32767),
    (q15_t)(1.0f  * 32767),
    (q15_t)(0.0f  * 32767),
    (q15_t)(-1.0f * 32767),
    (q15_t)(1.0f  * 32767),
    (q15_t)(0.0f  * 32767),
    (q15_t)(-1.0f * 32767),
		0,0,0
};

// --- Output (6x6) ------------------------------------------
q15_t pDst_q15[36] = {0};

// --- New MRAM addresses (different from float32!) ----------
uint32_t *pMRAM_Input_q15  = (uint32_t *)0x00083000;
uint32_t *pMRAM_Kernel_q15 = (uint32_t *)0x00084000;
uint32_t *pMRAM_Output_q15 = (uint32_t *)0x00085000;

// --- Write to MRAM -----------------------------------------
am_util_stdio_printf("Writing q15 input to MRAM...\n");
am_hal_mram_main_words_program(AM_HAL_MRAM_PROGRAM_KEY,
    (uint32_t *)pSrc_q15, pMRAM_Input_q15, 8);

am_util_stdio_printf("Writing q15 kernel to MRAM...\n");
am_hal_mram_main_words_program(AM_HAL_MRAM_PROGRAM_KEY,
    (uint32_t *)pKernel_q15, pMRAM_Kernel_q15, 8);

// --- Read back from MRAM -----------------------------------
am_util_stdio_printf("Reading back from MRAM...\n");
q15_t pSrc_read_q15[16];
q15_t pKernel_read_q15[9];

for (int i = 0; i < 8; i++)
{
    uint32_t word = *(pMRAM_Input_q15 + i);
    pSrc_read_q15[i*2]   = (q15_t)(word & 0xFFFF);
    pSrc_read_q15[i*2+1] = (q15_t)(word >> 16);
}

for (int i = 0; i < 4; i++)
{
    uint32_t word = *(pMRAM_Kernel_q15 + i);
    pKernel_read_q15[i*2] = (q15_t)(word & 0xFFFF);
    if (i*2+1 < 9)
        pKernel_read_q15[i*2+1] = (q15_t)(word >> 16);
}

// --- 2D Convolution ----------------------------------------
am_util_stdio_printf("Performing Q15 2D Convolution...\n\n");
for (int row = 0; row < 4; row++)
{
    arm_conv_q15(&pSrc_read_q15[row * 4], 4,
                  pKernel_read_q15, 3,
                  &pDst_q15[row * 6]);
}

// --- Write output to MRAM ----------------------------------
am_util_stdio_printf("Writing q15 output to MRAM...\n\n");
am_hal_mram_main_words_program(AM_HAL_MRAM_PROGRAM_KEY,
    (uint32_t *)pDst_q15, pMRAM_Output_q15, 18);

// --- Print output ------------------------------------------
am_util_stdio_printf("Q15 Convolution Output (6x6):\n");
for (int i = 0; i < 6; i++)
{
    for (int j = 0; j < 6; j++)
    {
        float32_t val = (float32_t)pDst_q15[i*6+j] / 32768.0f;
        am_util_stdio_printf("%6.4f ", val);
    }
    am_util_stdio_printf("\n");
}
am_util_stdio_printf("\nQ15 Done!\n");

    am_hal_uart_tx_flush(phUART);
    CHECK_ERRORS(am_hal_uart_power_control(phUART, AM_HAL_SYSCTRL_DEEPSLEEP, false));
		
    while (1)
    {
        am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
    }
}
