#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <fstream>
#include <string>
#include <limits>
#include <bits/stdc++.h>
#include <time.h>
#include <sys/resource.h>
#include "conv2d.h"

#define 	MEM_COMPL		2473936
#define		INPUT_IMAGE_DIM		224
#define		INPUT_MAP_SIZE		INPUT_IMAGE_DIM*INPUT_IMAGE_DIM*3

/*
 * tf.nn.conv2d prefers maxing out top left receptive field when conv'ing with input tile sizes that dont
 * completely fit the stride. In other words, if floor((W - k + 2*padding)/stride) != (W-k+2*padding)/stride,
 * conv2d has to start from (0, 0) (row, col) instead of the normal (-1, -1) for padding = 1.  
*/
/*
void conv2d(float *out, float *kernel, float *data, uint32_t K, uint8_t k, uint32_t W, uint32_t H, uint32_t C, uint8_t padding = 0, uint8_t stride = 1){

	uint32_t u, v, c, x, y;
	uint tilesR = floor((W - k + 2 * padding)/stride) + 1;
	uint tilesC = floor((H - k + 2 * padding)/stride) + 1;
	uint outSize = tilesR * tilesC;
	uint col = 0, row = 0, temp1, temp2;

	for(int kIter = 0; kIter < K; kIter++){
		for(y = 0; y < tilesC; y++){
			for(x = 0; x < tilesR; x++){
				out[kIter*tilesC*tilesR+tilesR*y + x] = 0;		//Clear output array
			}
		}
	}
	uint8_t uneven = float(tilesR - 1) != float(W-k+2*padding)/float(stride) ? 1 : 0;
	for(int kIter = 0; kIter < K; kIter++){		//Iterate over K kernels
		for(uint t = 0; t < outSize; t++){
			if(uneven){
				col = (t%tilesR)*stride;
				row = floor(t/tilesR)*stride;
			}else{
				col = (t % tilesR) * stride - padding;
				row = floor(t / tilesR) * stride - padding;
			}
//			printf("%d, %d\n", row, col);
				for(c = 0; c < C; c++){			//Iterate over Channels
					for(u = 0; u < k; u++){			//Iterate vertically within tile
						for(v = 0; v < k; v++){			//Iterate horizontally within tile
//							dataPoint = *(data + (c)*H*W + W*(y) + (x) + W*(u) + (v));
//							kernelPoint = *(kernel + (kIter)*(k*k*C) + (c)*k*k + k*(u) + (v));
//							outIndex = (kIter)*(W-k+1)*(H-k+1) + (W-k+1)*(y) + x;
							temp1 = row + u;
							temp2 = col + v;
//							printf("t: %d, %d, %d\n", t, row, col);
							if((temp1 < 0) || (temp2 < 0) || (temp1 >= H) || (temp2 >= W))
								*(out + kIter*outSize + t) += 0;
							else
								*(out + kIter*outSize + t) += (*(data + c*H*W + temp1*W + temp2)) * (*(kernel + kIter*k*k*C  + c*k*k + k*u + v));
//							printf("u = %d, v = %d, DP = %f, KP = %f, outIndex = %f \n", u, v, dataPoint, kernelPoint, out[outIndex]);
						}
					}
				}
		}
	}
}
*/

uint64_t t_pwgconv=0, t_dwconv=0, t_bn=0, t_relu=0, t_chsh=0, t_fc=0;
clock_t tSt;

void relu(float *activations, uint32_t W, uint32_t H, uint32_t C){
	tSt = clock();
	for(uint i = 0; i < W*H*C; i++){
		activations[i] = activations[i] < 0 ? 0 : activations[i];
	}
	t_relu += clock() - tSt;
}

void shufflenet_preprocess(float *activations, uint H, uint W, uint C){
//	float store[INPUT_MAP_SIZE];
	float *store = new float[INPUT_MAP_SIZE];
	for(int i = 0; i < C; i++){
		for(int j = 0; j < H; j++){
			for(int l = 0; l < W; l++){
				if(i == 0){
					store[2*H*W + j*W + l] = (activations[l+j*W+i*H*W] - 123.68) * 0.017;
				}else if(i == 1){
					store[H*W + j*W + l] = (activations[l+j*W+i*H*W] - 116.779) * 0.017;
				}else if(i == 2){
					store[j*W + l] = (activations[l+j*W+i*H*W] - 103.939) * 0.017;
				}
			}
		}
	}
	for(int i = 0; i < INPUT_MAP_SIZE; i++){
		activations[i] = store[i];
	}
	delete[] store;
}

void pool(float *out, float *data, uint8_t k, uint32_t H, uint32_t W, uint32_t C, uint8_t padding = 0, uint8_t stride = 1, uint8_t max_or_avg = 1){

	uint32_t u, v, x, y;
	float dataPoint;
	uint tilesR = floor((W - k + 2 * padding)/stride) + 1;
	uint tilesC = floor((H - k + 2 * padding)/stride) + 1;
	uint outSize = tilesR * tilesC;
	float avg_val = 0.0;
	float max_val = -99999.0;
	uint8_t valid_nums_avg = 0;
	uint col = 0, row = 0, temp1, temp2;

	for(uint kIter = 0; kIter < C; kIter++){
		for(y = 0; y < tilesC; y++){
			for(x = 0; x < tilesR; x++){
				out[kIter*tilesC*tilesR+tilesR*y+x] = 0;		//Clear output array
			}
		}
	}
	uint8_t uneven = float(tilesR - 1) != float(W-k+2*padding)/float(stride) ? 1 : 0;
	for(uint c = 0; c < C; c++){
		for(uint t = 0; t < outSize; t++){
			if(uneven){
				col = (t%tilesR)*stride;
				row = floor(t/tilesR)*stride;
			}else{
				col = (t % tilesR) * stride - padding;
				row = floor(t / tilesR) * stride - padding;
			}
					for(u = 0; u < k; u++){			//Iterate vertically within tile
						for(v = 0; v < k; v++){			//Iterate horizontally within tile
//							kernelPoint = *(kernel + (kIter)*(k*k*C) + (c)*k*k + k*(u) + (v));
//							outIndex = (kIter)*(W-k+1)*(H-k+1) + (W-k+1)*(y) + x;
							temp1 = row + u;
							temp2 = col + v;
//							printf("t: %d, %d, %d\n", t, row, col);
//							printf("%f ,\r\n", dataPoint);
							if((temp1 < 0) || (temp2 < 0) || (temp1 >= H) || (temp2 >= W)){
								if(max_val < 0 && max_or_avg == 1){
//									max_val = 0;
									continue;
								}
							}
							else{
								dataPoint = *(data + c*H*W + W*temp1 + temp2);
								if(max_val < dataPoint && max_or_avg == 1){
									max_val = dataPoint;
								}
								else if (max_or_avg == 0){
									avg_val += dataPoint;
									valid_nums_avg++;
								}
							}
//								*(out + kIter*outSize + t) += (*(data + c*H*W + temp1*W + temp2)) * (*(kernel + kIter*k*k*C  + c*k*k + k*u + v));
//							printf("u = %d, v = %d, DP = %f, KP = %f, outIndex = %f \n", u, v, dataPoint, kernelPoint, out[outIndex]);
						}
					}
					if(max_or_avg == 1){
						*(out+c*outSize+t) = max_val;
						max_val = -99999.0;
					}else if (max_or_avg == 0){
						*(out+c*outSize+t) = avg_val / float(valid_nums_avg);
						avg_val = 0.0, valid_nums_avg = 0;
					}
		}
	}
}

void depthwise_conv2d(float *out, float *kernel, float *data, uint8_t k, uint32_t W, uint32_t H, uint32_t C, uint8_t padding = 0, uint8_t stride = 1){
	tSt = clock();
	uint tilesR = floor((W - k + 2 * padding)/stride) + 1;
	uint tilesC = floor((H - k + 2 * padding)/stride) + 1;
	for (uint i = 0; i < C; i++){
		conv2d(&out[i*tilesR*tilesC], &kernel[i*k*k], &data[i*W*H], 1, k, W, H, 1, padding, stride);
	}
	t_dwconv += clock()-tSt;
}

void grouped_conv2d(float *out, float *kernel, float *data, uint32_t K, uint8_t k, uint32_t W, uint32_t H, uint32_t C, uint8_t num_grp){
	tSt = clock();
	uint tilesR = W, tilesC = H;
	uint ch_per_grp = C / num_grp;
	uint kern_per_grp = K / num_grp;
	for(uint i = 0; i < num_grp; i++)
		conv2d(&out[kern_per_grp*H*W*i], &kernel[ch_per_grp*kern_per_grp*i], &data[H*W*ch_per_grp*i], kern_per_grp, k, W, H, ch_per_grp, 0, 1);
	t_pwgconv += clock() - tSt;
}

void batch_normalization(float *activations, float *mean, float *variance, float* beta, float *gamma, uint H, uint W, uint C){
	tSt = clock();
	for(uint i = 0; i < C; i++){
		for(uint j = 0; j < H*W; j++){
			activations[j + i*H*W] = gamma[i] * (activations[j+i*H*W] - mean[i]) / sqrt(variance[i]+0.001) + beta[i];
		}
	}
	t_bn += clock() - tSt;
}

void channel_shuffle(float *out, float *activations, uint C, uint W, uint H, uint8_t num_grp){
	tSt = clock();
	uint ch_per_grp = C/num_grp;
	for (int i = 0; i < num_grp; i++){
		for (int j = 0; j < ch_per_grp; j++){
			for (int l = 0; l < W*H; l++){
				out[l + j*H*W + i*ch_per_grp*H*W] = activations[l + j*num_grp*H*W + i*H*W];
			}
		}
	}
	t_chsh += clock() - tSt;
}

uint64_t compute_kernel_offset(uint8_t stage, uint8_t block, uint8_t layer, uint8_t bn){
	if(stage == 1 && bn == 0)
		return 0;
	else if(stage == 1 && bn == 1)
		return 3*3*3*24;
	else if(stage == 2 && block == 0 && layer == 0 && bn == 0)
		return 3*3*3*24+24*4;
	else if(stage == 2 && block == 0 && layer == 0 && bn == 1)
		return 3*3*3*24+24*4+24*96;
	else if(stage == 2 && block == 0 && layer == 1 && bn == 0)
		return 3*3*3*24+24*4+24*96+96*4;
	else if(stage == 2 && block == 0 && layer == 1 && bn == 1)
		return 3*3*3*24+24*4+24*96+96*4+3*3*96;
	else if(stage == 2 && block == 0 && layer == 2 && bn == 0)
		return 3*3*3*24+24*4+24*96+96*4+3*3*96+96*4;
	else if(stage == 2 && block == 0 && layer == 2 && bn == 1)
		return 3*3*3*24+24*4+24*96+96*4+3*3*96+96*4+12*360;
	else if(stage == 2 && block > 0 && block < 4){
		if(layer == 0)
			return 10440 + (block - 1)*12384 + bn*48*96;
		else if(layer == 1)
			return 10440 + (block - 1)*12384 + 48*96+96*4 +bn*9*96;
		else if(layer == 2)
			return 10440 + (block - 1)*12384 + 48*96+96*4 + 9*96+96*4 + bn*12*384;
		else
			printf("Wrong input to function compute_kernel_offset\r\n");
	}
	else if(stage == 3 && block == 0){
		if(layer == 0)
			return 47592 + bn*48*192;
		else if(layer == 1)
			return 47592 + 48*192+192*4 + bn*9*192;
		else if(layer == 2)
			return 47592 + 48*192+192*4 + 9*192+192*4 + bn*24*384;
	}
	else if(stage == 3 && block > 0 && block < 8){
		if(layer == 0)
			return 70824 + (block - 1)*43200 + bn*96*192;
		else if(layer == 1)
			return 70824 + (block - 1)*43200 + 192*96 + 192*4 + bn*9*192;
		else if(layer == 2)
			return 70824 + (block - 1)*43200 + 192*96+192*4 + 9*192+192*4 + bn*24*768;
	}
	else if(stage == 4 && block == 0){
		if(layer == 0)
			return 373224 + bn*96*384;
		else if(layer == 1)
			return 373224 + 96*384+384*4 + bn*9*384;
		else if(layer == 2)
			return 373224 + 96*384+384*4 + 9*384+384*4 + bn*48*768;
	}
	else if(stage == 4 && block > 0 && block < 4){
		if(layer == 0)
			return 456552 + (block - 1)*160128 + bn*192*384;
		else if(layer == 1)
			return 456552 + (block - 1)*160128 + 192*384+384*4 + bn*9*384;
		else if(layer == 2)
			return 456552 + (block - 1)*160128 + 192*384+384*4 + 9*384+4*384 + bn*48*1536;
	}
	else if(stage == 5 && layer == 0)
		return 936936;
	else if(stage == 5 && layer == 1)
		return 2472936;

}

/*
 * Shufflenet Unit, Takes raw pointer to kernels vector, stage number between 1 and 4 inclusing,
 * block number, H, W, and C of input activations
*/

void shufflenet_unit(float *out_act, float *activations, float *kernels, uint8_t stage, uint8_t block, uint H, uint W, uint C, uint conv1_ch, uint conv1_K, uint conv2_ch, uint conv2_K, uint8_t stride, uint8_t num_groups = 8){
//	float residual_act[H*W*C];
//	float conv1_out[H*W*conv1_K];
//	float ch_shuffle_out[H*W*conv1_K];
//	float dconv_out[(H/stride)*(W/stride)*conv1_K];
//	float conv2_out[(H/stride) * (W/stride) * conv2_K];
//	float avg_pool_out[(H/stride)*(W/stride)*(C+conv2_K)];
	clock_t t_start, t_end;
	float *residual_act = new float[H*W*C];
	float *conv1_out = new float[H*W*conv1_K];
	float *ch_shuffle_out = new float[H*W*conv1_K];
	float *dconv_out = new float[(H/stride)*(W/stride)*conv1_K];
	float *conv2_out = new float[(H/stride) * (W/stride) * conv2_K];
	float *avg_pool_out = new float[(H/stride)*(W/stride)*(C+conv2_K)];
	for(uint i = 0; i < H*W*C; i++){
		residual_act[i] = activations[i];
	}
	uint64_t conv1_offset = compute_kernel_offset(stage, block, 0, 0);
	uint64_t conv1_bn_offset = compute_kernel_offset(stage, block, 0, 1);
	uint64_t dconv_offset = compute_kernel_offset(stage, block, 1, 0);
	uint64_t dconv_bn_offset = compute_kernel_offset(stage, block, 1, 1);
	uint64_t conv2_offset = compute_kernel_offset(stage, block, 2, 0);
	uint64_t conv2_bn_offset = compute_kernel_offset(stage, block, 2, 1);
	if(stage == 2 && block == 0){
		grouped_conv2d(conv1_out, kernels+conv1_offset, activations, conv1_K, 1, W, H, C, 1);
	}else{	
		grouped_conv2d(conv1_out, kernels+conv1_offset, activations, conv1_K, 1, W, H, C, 8);
	}
	batch_normalization(conv1_out, kernels+conv1_bn_offset, kernels+conv1_bn_offset+conv1_K, kernels+conv1_bn_offset+2*conv1_K, kernels+conv1_bn_offset+3*conv1_K, W, H, conv1_K);
	relu(conv1_out, W, H, conv1_K);
	channel_shuffle(ch_shuffle_out, conv1_out, conv1_K, W, H, 8);
	delete[] conv1_out;
	depthwise_conv2d(dconv_out, kernels+dconv_offset, ch_shuffle_out, 3, W, H, conv1_K, 1, stride);
	delete[] ch_shuffle_out;
	batch_normalization(dconv_out, kernels+dconv_bn_offset, kernels+dconv_bn_offset+conv1_K, kernels+dconv_bn_offset+2*conv1_K, kernels+dconv_bn_offset+3*conv1_K, W/stride, H/stride, conv1_K);	
	grouped_conv2d(conv2_out, kernels+conv2_offset, dconv_out, conv2_K, 1, W/stride, H/stride, conv1_K, 8);
	delete[] dconv_out;
	batch_normalization(conv2_out, kernels+conv2_bn_offset, kernels+conv2_bn_offset+conv2_K, kernels+conv2_bn_offset+2*conv2_K, kernels+conv2_bn_offset+3*conv2_K, W/stride, H/stride, conv2_K);
	if(stride == 1){
		for(uint i = 0; i < H*W*conv2_K; i++){
			conv2_out[i] += residual_act[i];
		}
		relu(conv2_out, W, H, conv2_K);
		for(uint i = 0; i < W*H*conv2_K; i++){
			out_act[i] = conv2_out[i];
		}
	}else if(stride == 2){
		pool(avg_pool_out, residual_act, 3, H, W, C, 1, stride, 0);
		for(uint i = (H/2)*(W/2)*C; i < (H/2)*(W/2)*(C+conv2_K); i++){
			avg_pool_out[i] = conv2_out[i-(H/2)*(W/2)*C]; //TODO: To perform tests, dont look at concatenated residual link!!!
		}
		relu(avg_pool_out, W/2, H/2, C+conv2_K);
		for(uint i = 0; i < (W/2)*(H/2)*(C+conv2_K); i++){
			out_act[i] = avg_pool_out[i];
		}
	}
	delete[] residual_act;
	delete[] conv2_out;
	delete[] avg_pool_out;
}

void shufflenet_stage1(float *out_act, float *in_act, float *kernels){
//	float conv1_out[112*112*24];
	clock_t t_start, t_end;
	float *conv1_out = new float[112*112*24];
	tSt = clock();
	conv2d(conv1_out, kernels, in_act, 24, 3, INPUT_IMAGE_DIM, INPUT_IMAGE_DIM, 3, 1, 2); 
	t_pwgconv += clock() - tSt;
	tSt = clock();
	uint64_t conv1_bn_offset = compute_kernel_offset(1, 0, 0, 1);	
	batch_normalization(conv1_out, kernels+conv1_bn_offset, kernels+conv1_bn_offset+24, kernels+conv1_bn_offset+48, kernels+conv1_bn_offset+72, 112, 112, 24);
	t_bn += clock() - tSt;
	pool(out_act, conv1_out, 3, INPUT_IMAGE_DIM/2, INPUT_IMAGE_DIM/2, 24, 1, 2, 1);
	delete[] conv1_out;
}

void shufflenet_stage(float *out_act, float *in_act, float *kernels, uint8_t stage, uint repeat, uint i_dim, uint i_ch, uint o_ch, uint conv1_C, uint conv1_K, uint conv2_C, uint conv2_K){
	

//	float blockx_out0[(i_dim/2)*(i_dim/2)*o_ch];
//	float blockx_out1[(i_dim/2)*(i_dim/2)*o_ch];
	float *blockx_out0 = new float[(i_dim/2)*(i_dim/2)*o_ch];
	float *blockx_out1 = new float[(i_dim/2)*(i_dim/2)*o_ch];
	shufflenet_unit(blockx_out1, in_act, kernels, stage, 0, i_dim, i_dim, i_ch, conv1_C, conv1_K, conv2_C, conv2_K, 2, 8);

	for(uint i = 1; i < repeat+1; i++){
		if(i%2 == 0){
			shufflenet_unit(blockx_out1, blockx_out0, kernels, stage, i, i_dim/2, i_dim/2, o_ch, conv1_C*2, conv1_K, conv2_C, o_ch, 1, 8);
		}else{
			shufflenet_unit(blockx_out0, blockx_out1, kernels, stage, i, i_dim/2, i_dim/2, o_ch, conv1_C*2, conv1_K, conv2_C, o_ch, 1, 8);
		}
	}
	for(uint i = 0; i < (i_dim/2)*(i_dim/2)*o_ch; i++){
		out_act[i] = blockx_out0[i];
	}
	delete[] blockx_out0;
	delete[] blockx_out1;
}

void fc_layer(float *out_act, float *in_act, float *kernels, uint i_ch, uint o_ch){
	tSt = clock();
	uint64_t w_offset = compute_kernel_offset(5, 0, 0, 0);
	uint64_t b_offset = compute_kernel_offset(5, 0, 1, 0);
	for(uint i = 0; i < o_ch; i++){
		out_act[i] = 0;
	}
	for(uint i = 0; i < o_ch; i++){
		for(uint j = 0; j < i_ch; j++){
			out_act[i] += in_act[j] * kernels[w_offset+o_ch*j+i];
		}
		out_act[i] += kernels[b_offset+i];
	}
	t_fc += clock() - tSt;
}

void softmax_unit(float *activations, uint dim){
	float exp_sum = 0;
	for(uint i = 0; i < dim; i++){
		activations[i] = exp(activations[i]);
		exp_sum += activations[i];
	}
	for(uint i = 0; i < dim; i++){
		activations[i] = activations[i] / exp_sum;
	}
}

int main(void){
	printf("Allocating Memory...\n");
//	float kernel_vec[MEM_COMPL];// = new float[MEM_COMPL];
//	float feature_vec[INPUT_MAP_SIZE];// = new float[INPUT_MAP_SIZE];
//	float stage1_out[(INPUT_IMAGE_DIM/4)*(INPUT_IMAGE_DIM/4)*24];// = new float[(INPUT_IMAGE_DIM/4)*(INPUT_IMAGE_DIM/4)*24];
//	float stage2_out[(INPUT_IMAGE_DIM/8)*(INPUT_IMAGE_DIM/8)*384];// = new float[(INPUT_IMAGE_DIM/8)*(INPUT_IMAGE_DIM/8)*384];
//	float stage3_out[(INPUT_IMAGE_DIM/16)*(INPUT_IMAGE_DIM/16)*768];// = new float[(INPUT_IMAGE_DIM/16)*(INPUT_IMAGE_DIM/16)*768];
//	float stage4_out[(INPUT_IMAGE_DIM/32)*(INPUT_IMAGE_DIM/32)*1536];// = new float[(INPUT_IMAGE_DIM/32)*(INPUT_IMAGE_DIM/32)*1536];
//	float g_pool_out[1536];// = new float[1536];
//	float fc_out[1000];// = new float[1000];

	float *kernel_vec = new float[MEM_COMPL];
	float *feature_vec = new float[INPUT_MAP_SIZE];
	float *stage1_out = new float[(INPUT_IMAGE_DIM/4)*(INPUT_IMAGE_DIM/4)*24];
	float *stage2_out = new float[(INPUT_IMAGE_DIM/8)*(INPUT_IMAGE_DIM/8)*384];
	float *stage3_out = new float[(INPUT_IMAGE_DIM/16)*(INPUT_IMAGE_DIM/16)*768];
	float *stage4_out = new float[(INPUT_IMAGE_DIM/32)*(INPUT_IMAGE_DIM/32)*1536];
	float *g_pool_out = new float[1536];
	float *fc_out = new float[1000];

	std::fstream image("./image.bin");
	std::fstream model("./shufflenet-model-1x-8g.bin");

	std::string str0;
	printf("Importing image_bin...\n");
	for (int i = 0; i < INPUT_MAP_SIZE; i++){
		image >> str0;
		feature_vec[i] = std::stof(str0.c_str());
//		feature_vec[i] = float(i)/float(INPUT_MAP_SIZE);
	}
	printf("image_bin imported...\n");
	printf("Importing kernel_vec...\n");
	for (int i = 0; i < MEM_COMPL; i++){
		model >> str0;
		kernel_vec[i] = std::stof(str0.c_str());
//		kernel_vec[i] = float(i)/float(MEM_COMPL);
	}
	printf("kernel_vec imported...\n");
	clock_t start, end, t_net_s, t_net_e;
	double cpu_time_used;
#if	USE_CBLAS
	printf("Computing GEMMs using CBLAS\r\n");
#else
	printf("Computing GEMMs using Loops\r\n");
#endif
for(uint i = 0; i < 50; i++){
	start = clock();
	t_net_s = clock();
	shufflenet_preprocess(feature_vec, INPUT_IMAGE_DIM, INPUT_IMAGE_DIM, 3);
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Preprocessing done in %f Seconds\r\n", cpu_time_used);
	start = clock();
	shufflenet_stage1(stage1_out, feature_vec, kernel_vec);
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("stage 1 done in %f Seconds\r\n", cpu_time_used);
	start = clock();
	shufflenet_stage(stage2_out, stage1_out, kernel_vec, 2, 3, 56, 24, 384, 24, 96, 12, 360); 
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("stage 2 done in %f Seconds\r\n", cpu_time_used);
	start = clock();
	shufflenet_stage(stage3_out, stage2_out, kernel_vec, 3, 7, 28, 384, 768, 48, 192, 24, 384);
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("stage 3 done in %f Seconds\r\n", cpu_time_used);
	start = clock();
	shufflenet_stage(stage4_out, stage3_out, kernel_vec, 4, 3, 14, 768, 1536, 96, 384, 48, 768);
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("stage 4 done in %f Seconds\r\n", cpu_time_used);
	start = clock();
	pool(g_pool_out, stage4_out, 7, 7, 7, 1536, 0, 1, 0);
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("G Pool done in %f Seconds\r\n", cpu_time_used);
	start = clock();
	fc_layer(fc_out, g_pool_out, kernel_vec, 1536, 1000);
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("FC layer done in %f Seconds\r\n", cpu_time_used);
	start = clock();
	softmax_unit(fc_out, 1000);
	end = clock();
	t_net_e = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Softmax done in %f Seconds\r\n", cpu_time_used);
	printf("Net time: %f Seconds\r\n", ((double)(t_net_e - t_net_s))/CLOCKS_PER_SEC);
	std::sort(fc_out, fc_out+1000, std::greater<float>());
	printf("Printing Top-5 probabilities\n");
	for(int i = 0; i < 5; i++)
		printf("%f, ", fc_out[i]);
	printf("\n");
	printf("t_pwgconv = %f, t_dconv = %f, t_bn = %f, t_relu = %f, t_chsh = %f, t_fc = %f\r\n", (double)(t_pwgconv)/CLOCKS_PER_SEC, (double)(t_dwconv)/CLOCKS_PER_SEC, (double)(t_bn)/CLOCKS_PER_SEC, (double)(t_relu)/CLOCKS_PER_SEC, (double)(t_chsh)/CLOCKS_PER_SEC, (double)(t_fc)/CLOCKS_PER_SEC);
}
	delete[] kernel_vec;
	delete[] stage1_out;
	delete[] stage2_out;
	delete[] stage3_out;
	delete[] stage4_out;
	delete[] fc_out;
	delete[] feature_vec;
	return 0;
}
/*
int main(void){
	const rlim_t kStackSize = 32 * 1024 * 1024;   // min stack size = 16 MB
	struct rlimit rl;
	int result;

	result = getrlimit(RLIMIT_STACK, &rl);
	if (result == 0)
	{
		if (rl.rlim_cur < kStackSize){
			rl.rlim_cur = kStackSize;
			result = setrlimit(RLIMIT_STACK, &rl);
			if (result != 0){
				fprintf(stderr, "setrlimit returned result = %d\n", result);
			}
		}
	}
	shufflenet_run();
	return 0;

}*/
