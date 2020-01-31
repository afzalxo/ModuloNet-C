#include "matmul.h"
#include "layer_ops.h"
#include "utils.h"

void sign(uint32_t dim0, uint32_t dim1, int16_t inp[][dim1], int16_t res[][dim1]){
	for (uint32_t i = 0; i < dim0; i++){
		for(uint32_t j = 0; j < dim1; j++){
				res[i][j] = (inp[i][j] > 0) ? 1 : -1;
		}
	}
}

//Implements activation function y(x) = -sign(x-k/2)
void activation_fn(uint32_t dim0, uint32_t dim1, uint16_t k_val, uint16_t inp[][dim1], int16_t res[][dim1]){
	int16_t (*temp0)[dim1] = malloc(sizeof(int16_t[dim0][dim1]));
	for (uint32_t i = 0; i < dim0; i++){
		for (uint32_t j = 0; j < dim1; j++){
			temp0[i][j] = inp[i][j] - (k_val/2);	
		}
	}
	sign(dim0, dim1, temp0, res);
	for (uint32_t i = 0; i < dim0; i++){
		for (uint32_t j = 0; j < dim1; j++){
			res[i][j] = -1*res[i][j];
		}
	}
	free(temp0);
}
//Addition of biases into the second dimension of the input feature map. The first dimension is the batch dim. Pass num_nodes as dim1
void _bias_add(uint32_t dim0, uint32_t dim1, int16_t inp[][dim1], int16_t biases[], int16_t res[][dim1]){
	for (uint32_t i = 0; i < dim0; i++){
		for (uint32_t j = 0; j < dim1; j++){
			res[i][j] = inp[i][j] + biases[j];
		}
	}
}

//Implements y=xw + b where both x and w are matrices. w is a bin matrix 
//while x is bin for all layers except layer 0.
void bin_dense_layer(uint32_t inp_dim0, uint32_t inp_dim1, uint32_t num_nodes, int16_t inp[][inp_dim1], int16_t in_w[][num_nodes], int16_t in_b[], int16_t layer_out[][num_nodes]){
	
	int16_t (*xw_out)[num_nodes] = malloc(sizeof(int16_t[inp_dim0][num_nodes]));

	matmul_int16_int16(inp_dim0, inp_dim1, inp_dim1, num_nodes, inp, in_w, xw_out);
	_bias_add(inp_dim0, num_nodes, xw_out, in_b, layer_out);

	free(xw_out);
}

//It seems that % operator in C99 is a remainder operator and not the modulo we need.
//Behaviour of % is different from tf.floormod operation for negative inputs.
//Please see the following link to try and understand. 
//https://stackoverflow.com/questions/11720656/modulo-operation-with-negative-numbers
void mod_layer(uint32_t dim0, uint32_t dim1, uint16_t k_val, int16_t inp[][dim1], uint16_t out[][dim1]){
	int32_t r = 0;
	for (uint32_t i = 0; i < dim0; i++){
		for (uint32_t j = 0; j < dim1; j++){
			r = inp[i][j] % k_val;
			out[i][j] = (r < 0) ? r + k_val : r;
			//out[i][j] = inp[i][j] % k_val;
		}
	}
}

//Testbed for bin_dense_layer
/*
void main(void){

	int16_t inp0[1][4] = {{2, 3, 13, 23}};
	int16_t in_w0[4][4] = {{-1, -4, 2, 8},{1, 2, 6, 4},{2, -3, -2, 1},{-3, 1, 1, -9}};
	int16_t in_b0[4] = {3, 4, -9, 2}; //{3, 4, -9, 2}
	int16_t layer_o[1][4];
	bin_dense_layer(1, 4, 4, inp0, in_w0, in_b0, layer_o);
	printmat_int16(1, 4, layer_o);
}
*/
