#include "modulonet.h"
#include "globals.h"

//Testbench modulonet_mlp
void main(void){
	int16_t (*inp)[784] = malloc(sizeof(int16_t[NUM_IMAGES][784]));
	load_input_images(NUM_IMAGES, inp);
	//printmat_int16(1, 784, inp);
	modulonet_mlp(NUM_IMAGES, 784, inp);
}
