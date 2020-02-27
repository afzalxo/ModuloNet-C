#ifndef MODULONET_H
#define MODULONET_H
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include "utils.h"
#include "layer_ops.h"

void load_pretrained_model(int16_t w0[784][4096], int16_t w1[4096][4096], int16_t w2[4096][4096], int16_t w3[4096][10], int16_t b0[784], int16_t b1[4096], int16_t b2[4096], int16_t b3[10]);
void modulonet_mlp(uint32_t in_dim0, uint32_t in_dim1, int16_t inp[][in_dim1]);


#endif
