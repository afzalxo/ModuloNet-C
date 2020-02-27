#ifndef _CONV2D_
#define _CONV2D_

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <cblas.h>
#include <assert.h>

#define 	USE_CBLAS	1

inline void _cblas_sgemm_stub(float *i_mat0, float *i_mat1, float *o_mat, uint mat0_dim0, uint mat0_dim1, uint mat1_dim0, uint mat1_dim1);

void conv2d(float *out, float *kernel, float *data, uint32_t K, uint8_t k, uint32_t W, uint32_t H, uint32_t C, uint8_t padding = 0, uint8_t stride = 1);

#endif
