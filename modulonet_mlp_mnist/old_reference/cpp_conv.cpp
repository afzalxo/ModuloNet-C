#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

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

	for(int kIter = 0; kIter < K; kIter++){		//Iterate over K kernels
		for(uint t = 0; t < outSize; t++){
			col = (t % tilesR) * stride - padding;
			row = floor(t / tilesR) * stride - padding;
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


void pool(float *out, float *data, uint8_t k, uint32_t W, uint32_t H, uint32_t C, uint8_t padding = 0, uint8_t stride = 1, uint8_t max_or_avg = 1){

	uint32_t u, v, x, y;
	float dataPoint;
//	float dataPoint, kernelPoint;
	uint tilesR = floor((W - k + 2 * padding)/stride) + 1;
	uint tilesC = floor((H - k + 2 * padding)/stride) + 1;
	uint outSize = tilesR * tilesC;
	float avg_val = 0.0;
	float max_val = -99999.0;
	uint8_t valid_nums_avg = 0;
	//	printf("Tiles: %d, %d, %d\n", tilesC, tilesR, outSize);
	uint col = 0, row = 0, temp1, temp2;

	for(uint kIter = 0; kIter < C; kIter++){
		for(y = 0; y < tilesC; y++){
			for(x = 0; x < tilesR; x++){
				out[kIter*tilesC*tilesR+tilesR*y+x] = 0;		//Clear output array
			}
		}
	}
	for(uint c = 0; c < C; c++){
		for(uint t = 0; t < outSize; t++){
			col = (t % tilesR) * stride - padding;
			row = floor(t / tilesR) * stride - padding;

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
									max_val = 0;
								}
							}
							else{
								dataPoint = *(data + c*H*W + W*temp1 + temp2);
								if(max_val < dataPoint && max_or_avg == 1)
									max_val = dataPoint;
								else if (max_or_avg == 0)
									avg_val += dataPoint;
									valid_nums_avg++;
							}
//								*(out + kIter*outSize + t) += (*(data + c*H*W + temp1*W + temp2)) * (*(kernel + kIter*k*k*C  + c*k*k + k*u + v));
//							printf("u = %d, v = %d, DP = %f, KP = %f, outIndex = %f \n", u, v, dataPoint, kernelPoint, out[outIndex]);
						}
					}
					if(max_or_avg == 1){
						*(out+c*outSize+t) = max_val;
						max_val = -999999;
					}else if (max_or_avg == 0){
						*(out+c*outSize+t) = avg_val / float(valid_nums_avg);
						avg_val = 0.0, valid_nums_avg = 0;
					}
		}
	}
}

void depthwise_conv2d(float *out, float *kernel, float *data, uint8_t k, uint32_t W, uint32_t H, uint32_t C, uint8_t padding = 0, uint8_t stride = 1){
	uint tilesR = floor((W - k + 2 * padding)/stride) + 1;
	uint tilesC = floor((H - k + 2 * padding)/stride) + 1;
	for (uint i = 0; i < C; i++){
		conv2d(&out[i*tilesR*tilesC], &kernel[i*k*k], &data[i*W*H], 1, k, W, H, 1, padding, stride);
	}
}

void grouped_conv2d(float *out, float *kernel, float *data, uint32_t K, uint8_t k, uint32_t W, uint32_t H, uint32_t C, uint8_t num_grp){
	uint tilesR = W, tilesC = H;
	uint ch_per_grp = C / num_grp;
	uint kern_per_grp = K / num_grp;
	for(uint i = 0; i < num_grp; i++)
		conv2d(&out[kern_per_grp*H*W*i], &kernel[ch_per_grp*kern_per_grp*i], &data[H*W*ch_per_grp*i], kern_per_grp, k, W, H, ch_per_grp, 0, 1);
}

void batch_normalization(float *out, float *activations, float *mean, float *variance, float* beta, float *gamma, uint W, uint H, uint C){
	for(uint i = 0; i < C; i++){
		for(uint j = 0; j < H*W; j++){
			out[j + i*H*W] = gamma[i] * (activations[j+i*H*W] - mean[i]) / sqrt(variance[i]+0.001) + beta[i];
		}
	}
}

void channel_shuffle(float *out, float *activations, uint C, uint W, uint H, uint8_t num_grp){
	uint ch_per_grp = C/num_grp;
	for (int i = 0; i < num_grp; i++){
		for (int j = 0; j < ch_per_grp; j++){
			for (int l = 0; l < W*H; l++){
				out[l + j*H*W + i*ch_per_grp*H*W] = activations[l + j*num_grp*H*W + i*H*W];
			}
		}
	}
}

int main(void){
	uint stride = 2, padding = 1;
	uint act_size = 3, ch = 16, K = 128, k = 1;
	uint out_tile_size = ((act_size-k+2*padding)/stride+1);
	float in_map[act_size *act_size *ch];
	float k_map[k*k*ch*K];
	float o_vec[K*out_tile_size*out_tile_size];
	float o_pool_vec[ch*out_tile_size*out_tile_size];
	float o_vec_dwconv[ch*out_tile_size*out_tile_size];
	float o_vec_gconv[K*act_size*act_size];
	float o_vec_bn[ch*act_size*act_size];
	float o_vec_ch_shuffle[ch*act_size*act_size];
	float mean_vec[ch], variance_vec[ch], beta_vec[ch], gamma_vec[ch];

	for(int i = 0; i < act_size*act_size*ch; i++){
		in_map[i] = float(i) / float(act_size*act_size*ch);
	}
	for(int i = 0; i < k*k*(ch/8)*K; i++){
		k_map[i] = float(i) / float(k*k*(ch/8)*K);
	}
	for(int i = 0; i < ch; i++){
		mean_vec[i] = i / float(ch);
		gamma_vec[i] = i / float(ch);
		variance_vec[i] = i / float(ch);
		beta_vec[i] = i / float(ch);
	}
//	conv2d(o_vec, k_map, in_map, K, k, act_size, act_size, ch, padding, stride);
//	pool(o_pool_vec, in_map, k, act_size, act_size, ch, padding, stride, 0);
//	depthwise_conv2d(o_vec_dwconv, k_map, in_map, k, act_size, act_size, ch, padding, stride);
//	grouped_conv2d(o_vec_gconv, k_map, in_map, K, 1, act_size, act_size, ch, 8);
//	batch_normalization(o_vec_bn, in_map, mean_vec, variance_vec, beta_vec, gamma_vec, act_size, act_size, ch); 
	channel_shuffle(o_vec_ch_shuffle, in_map, ch, act_size, act_size, 8);
	printf("\r\n");
//	for (int i = 0; i < K*out_tile_size*out_tile_size; i++)
//		printf("%f, ", o_vec[i]);
//	for (int i = 0; i < ch*out_tile_size*out_tile_size; i++){
//		printf("%f, ", o_pool_vec[i]);
//		printf("%f, ", o_vec_dwconv[i]);
//	}
//	for (int i = 0; i < K*act_size*act_size; i++){
//		printf("%f, ", o_vec_gconv[i]);
//	}
	for (int i = 0; i < ch*act_size*act_size; i++){
//		printf("%f, ", o_vec_bn[i]);
		printf("%f, ", o_vec_ch_shuffle[i]);
	}
	printf("\r\n");
	return 0;
}

