#ifndef MATMUL_H
#define MATMUL_H
#include <stdint.h>

void matmul_int16_int16(uint32_t mat0_dim0, uint32_t mat0_dim1, uint32_t mat1_dim0, uint32_t mat1_dim1, int16_t mat0[][mat0_dim1], int16_t mat1[][mat1_dim1], int16_t out_mat[][mat1_dim1]);

#endif

