#include "modulonet.h"
#include "globals.h"

//Loads weights and biases into the variables pointed to by the input pointers
void load_pretrained_model(int16_t w0[784][4096], int16_t w1[4096][4096], int16_t w2[4096][4096], int16_t w3[4096][10], int16_t b0[784], int16_t b1[4096], int16_t b2[4096], int16_t b3[10]){
	FILE *w0_f = fopen("l0_w.txt", "r");
	for (uint32_t i = 0; i < 4096; i++){
		for (uint32_t j = 0; j < 784; j++){
			fscanf(w0_f, "%" SCNd16, &w0[j][i]);
		}
	}
	fclose(w0_f);
	FILE *w1_f = fopen("l1_w.txt", "r");
	for (uint32_t i = 0; i < 4096; i++){
		for (uint32_t j = 0; j < 4096; j++){
			fscanf(w1_f, "%" SCNd16, &w1[j][i]);
		}
	}
	fclose(w1_f);
	FILE *w2_f = fopen("l2_w.txt", "r");
	for (uint32_t i = 0; i < 4096; i++){
		for (uint32_t j = 0; j < 4096; j++){
			fscanf(w2_f, "%" SCNd16, &w2[j][i]);
		}
	}
	fclose(w2_f);
	FILE *w3_f = fopen("l3_w.txt", "r");
	for (uint32_t i = 0; i < 10; i++){
		for (uint32_t j = 0; j < 4096; j++){
			fscanf(w3_f, "%" SCNd16, &w3[j][i]);
		}
	}
	fclose(w3_f);
	FILE *biases_fptr = fopen("biases.txt", "r");
	for (uint32_t i = 0; i < 4096; i++){
		fscanf(biases_fptr, "%" SCNd16, &b0[i]);
	}
	for (uint32_t i = 0; i < 4096; i++){
		fscanf(biases_fptr, "%" SCNd16, &b1[i]);
	}
	for (uint32_t i = 0; i < 4096; i++){
		fscanf(biases_fptr, "%" SCNd16, &b2[i]);
	}
	for (uint32_t i = 0; i < 10; i++){
		fscanf(biases_fptr, "%" SCNd16, &b3[i]);
	}
	fclose(biases_fptr);
}


void modulonet_mlp(uint32_t in_dim0, uint32_t in_dim1, int16_t inp[][in_dim1]){
	uint16_t k0 = 32768, k1 = 1024, k2 = 1024, k3 = 8192;
	//Allocating memory on heap for storing trained model
	int16_t (*w0_ptr)[4096] = malloc(sizeof(int16_t[784][4096]));
	int16_t (*w1_ptr)[4096] = malloc(sizeof(int16_t[4096][4096]));
	int16_t (*w2_ptr)[4096] = malloc(sizeof(int16_t[4096][4096]));
	int16_t (*w3_ptr)[10] = malloc(sizeof(int16_t[4096][10]));
	int16_t (*b0_ptr) = malloc(sizeof(int16_t)*4096);
	int16_t (*b1_ptr) = malloc(sizeof(int16_t)*4096);
	int16_t (*b2_ptr) = malloc(sizeof(int16_t)*4096);
	int16_t (*b3_ptr) = malloc(sizeof(int16_t)*10);

	load_pretrained_model(w0_ptr, w1_ptr, w2_ptr, w3_ptr, b0_ptr, b1_ptr, b2_ptr, b3_ptr);
	
	//Allocating memory for intermediate fn in/outs
	int16_t (*base_out0)[4096] = malloc(sizeof(int16_t[NUM_IMAGES][4096]));
	int16_t (*base_out2)[4096] = malloc(sizeof(int16_t[NUM_IMAGES][4096]));
	uint16_t (*base_out1)[4096] = malloc(sizeof(int16_t[NUM_IMAGES][4096]));
	int16_t (*fin_dense_out)[10] = malloc(sizeof(int16_t[NUM_IMAGES][10]));
	uint16_t (*fin_mod_out)[10] = malloc(sizeof(uint16_t[NUM_IMAGES][10]));
	uint16_t (*fin_o) = malloc(sizeof(uint16_t[NUM_IMAGES]));

	bin_dense_layer(NUM_IMAGES, 784, 4096, inp, w0_ptr, b0_ptr, base_out0); 
	mod_layer(NUM_IMAGES, 4096, k0, base_out0, base_out1);
	activation_fn(NUM_IMAGES, 4096, k0, base_out1, base_out2);

	bin_dense_layer(NUM_IMAGES, 4096, 4096, base_out2, w1_ptr, b1_ptr, base_out0); 
	mod_layer(NUM_IMAGES, 4096, k1, base_out0, base_out1);
	activation_fn(NUM_IMAGES, 4096, k1, base_out1, base_out2);
	
	bin_dense_layer(NUM_IMAGES, 4096, 4096, base_out2, w2_ptr, b2_ptr, base_out0); 
	mod_layer(NUM_IMAGES, 4096, k2, base_out0, base_out1);
	activation_fn(NUM_IMAGES, 4096, k2, base_out1, base_out0);

	bin_dense_layer(NUM_IMAGES, 4096, 10, base_out0, w3_ptr, b3_ptr, fin_dense_out);
	mod_layer(NUM_IMAGES, 10, k3, fin_dense_out, fin_mod_out);
	//printmat_uint16(NUM_IMAGES, 10, fin_mod_out);
	modulo_argmax(NUM_IMAGES, 10, k3, fin_mod_out, fin_o);

	printf("Test Output Class(es):\n");
	for (uint8_t i = 0; i < NUM_IMAGES; i++)
		printf("%" PRId16 " ", fin_o[i]);
	printf("\n");

	free(base_out0);
	free(base_out1);
	free(base_out2);
	free(fin_dense_out);
	free(fin_mod_out);

	free(w0_ptr);
	free(w1_ptr);
	free(w2_ptr);
	free(w3_ptr);
	free(b0_ptr);
	free(b1_ptr);
	free(b2_ptr);
	free(b3_ptr);
}

