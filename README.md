# Namuunaa_Thesis

## Environment
| Tool | Details |
|------|---------|
| Board | Ambiq Apollo4 Blue Lite (AMA4B2KP-KBR) |
| IDE | Keil uVision5 |
| Debugger | SEGGER J-Link |
| SDK | AmbiqSuite R4.5.0 |
| Terminal | PuTTY (COM3, 115200 baud) |
| Library | CMSIS-DSP (arm_cortexM4lf_math.lib) |

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
