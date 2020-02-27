#ifndef LAYER_OPS_H
#define LAYER_OPS_H
#include "globals.h"

void sign(uint32_t dim0, uint32_t dim1, int16_t inp[][dim1], int16_t res[][dim1]);
void activation_fn(uint32_t dim0, uint32_t dim1, uint16_t k_val, uint16_t inp[][dim1], int16_t res[][dim1]);
void _bias_add(uint32_t dim0, uint32_t dim1, int16_t inp[][dim1], int16_t biases[], int16_t res[][dim1]);
void bin_dense_layer(uint32_t inp_dim0, uint32_t inp_dim1, uint32_t num_nodes, int16_t inp[][inp_dim1], int16_t in_w[][num_nodes], int16_t in_b[], int16_t layer_out[][num_nodes]);
void mod_layer(uint32_t dim0, uint32_t dim1, uint16_t k_val, int16_t inp[][dim1], uint16_t out[][dim1]);

#endif

