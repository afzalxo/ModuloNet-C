#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "../includes/matmul.h"

//Multiply two matrices mat0 and mat1
void matmul_int16_int16(uint32_t mat0_dim0, uint32_t mat0_dim1, uint32_t mat1_dim0, uint32_t mat1_dim1, int16_t mat0[][mat0_dim1], int16_t mat1[][mat1_dim1], int16_t out_mat[][mat1_dim1]){
	assert(mat0_dim1 == mat1_dim0);
	//Clear memory locations of the resultant matrix
	for (uint32_t res_row = 0; res_row < mat0_dim0; res_row++){
		for (uint32_t res_col = 0; res_col < mat1_dim1; res_col++){
			out_mat[res_row][res_col] = 0;
		}
	}

	//Apply loop based matrix multiplication
	for (uint32_t res_row = 0; res_row < mat0_dim0; res_row++){
		for (uint32_t res_col = 0; res_col < mat1_dim1; res_col++){
			for (uint32_t col_iter = 0; col_iter < mat0_dim1; col_iter++){
				out_mat[res_row][res_col] += mat0[res_row][col_iter] * mat1[col_iter][res_col];
			}
		}
	}
}


//Testbed for matmul
/*
int main(void){
	int16_t mat0[3][3] = {{1, -3, 5},{2, 4, -8},{9, -12, 36}};
	int16_t mat1[3][3] = {{1, 9, -19},{2, 64, -33},{-259, 309, 316}};
	int16_t res_mat[3][3];
	matmul_int16_int16(3, 3, 3, 3, mat0, mat1, res_mat);
	printmat_int16(3, 3, res_mat);	
	return 0;
}
*/
