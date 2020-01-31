#ifndef UTILS_H
#define UTILS_H
#include "globals.h"

void load_input_images(uint32_t num_imgs, int16_t imgs[][784]);
void modulo_argmax(uint32_t dim0, uint32_t dim1, uint16_t modk, uint16_t inp[][dim1], uint16_t out[]);
void printmat_int16(uint32_t dim0, uint32_t dim1, int16_t mat[][dim1]);
void printmat_uint16(uint32_t dim0, uint32_t dim1, uint16_t mat[][dim1]);

#endif

