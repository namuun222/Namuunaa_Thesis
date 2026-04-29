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

## Dependencies
- AmbiqSuite SDK R4.5.0 (download from Ambiq website)
- Keil MDK with Cortex-M4 support
- SEGGER J-Link drivers
