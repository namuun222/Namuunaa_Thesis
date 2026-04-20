#ifndef NVM_CONFIG_H
#define NVM_CONFIG_H

#include <stdint.h>
#include <stdbool.h>

bool nvm_read(void);
bool nvm_write(void);

#endif