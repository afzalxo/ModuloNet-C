#include "utils.h"

void load_input_images(uint32_t num_imgs, int16_t imgs[][784]){
	FILE *img_f = fopen("mnist_int_images.csv", "r");
	if(img_f == NULL){
		printf("Error opening mnist_int_images.csv\r\n");
	}
	for (uint32_t i = 0; i < num_imgs; i++){
		for (uint32_t j = 0; j < 784; j++){
			fscanf(img_f, "%" SCNd16 ",", &imgs[i][j]);
		}
	}
	fclose(img_f);
}

//Print an int16_t matrix. Please note it cannot print a subtile of a matrix but only a full matrix. If dim0 and dim1 are passed as subtile lengths, the output would be wrong.
void printmat_int16(uint32_t dim0, uint32_t dim1, int16_t mat[][dim1]){
	for (uint32_t d0 = 0; d0 < dim0; d0++){
		for (uint32_t d1 = 0; d1 < dim1; d1++){
			printf("%d ", mat[d0][d1]);
		}
		printf("\n");
	}
}

void printmat_uint16(uint32_t dim0, uint32_t dim1, uint16_t mat[][dim1]){
	for (uint32_t d0 = 0; d0 < dim0; d0++){
		for (uint32_t d1 = 0; d1 < dim1; d1++){
			printf("%d ", mat[d0][d1]);
		}
		printf("\n");
	}
}

void modulo_argmax(uint32_t dim0, uint32_t dim1, uint16_t modk, uint16_t inp[][dim1], uint16_t out[]){
	int cur_modmax = -1;
	int modmax_ind = -1;
	for (uint32_t i = 0; i < dim0; i++){
		for (uint32_t j = 0; j < dim1; j++){
			if ((inp[i][j] > cur_modmax) && (inp[i][j] < modk/2)){
				cur_modmax = inp[i][j];
				modmax_ind = j;
			}
		}
		if(cur_modmax != -1){
			out[i] = modmax_ind;
		} else {
			for (uint32_t j = 0; j < dim1; j++){
				if((inp[i][j] > cur_modmax) && (inp[i][j] >= modk/2)){
					cur_modmax = inp[i][j];
					modmax_ind = j;
				}
			}
			out[i] = modmax_ind;
		}
		cur_modmax = -1;
		modmax_ind = -1;
	}
}

