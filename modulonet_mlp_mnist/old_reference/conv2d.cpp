#include "conv2d.h"

inline void _cblas_sgemm_stub(float *i_mat0, float *i_mat1, float *o_mat, uint mat0_dim0, uint mat0_dim1, uint mat1_dim0, uint mat1_dim1){
//	clock_t st = clock();
	assert (mat0_dim1 == mat1_dim0);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mat0_dim0, mat1_dim1, mat0_dim1, 1.0, i_mat0, mat0_dim1, i_mat1, mat1_dim1, 0.0, o_mat, mat1_dim1);
//	printf("gemm with %dx%d, %dx%d took %d cc\r\n", mat0_dim0, mat0_dim1, mat1_dim0, mat1_dim1, (clock()-st));
}

void conv2d(float *out, float *kernel, float *data, uint32_t K, uint8_t k, uint32_t W, uint32_t H, uint32_t C, uint8_t padding, uint8_t stride){
	uint out_rows = uint((H + 2*padding - k) / stride) + 1;
	uint out_cols = uint((W + 2*padding - k) / stride) + 1;
	uint temp0 = 0, temp1 = 0;
	float *reshaped_filters = new float[k*k*C*K];
	float *shards = new float[out_rows*out_cols*k*k*C];
#if	USE_CBLAS
	float *_cblas_gemm_out = new float[out_rows*out_cols*K];
#endif
	uint8_t uneven = float(out_rows - 1) != float(W-k+2*padding)/float(stride) ? 1 : 0;
//	clock_t t_net_s, t_net_e;
//	t_net_s = clock();
	//Reshaping filters from [k,k,C,K] to [k*k*C, K]
	for(uint c = 0; c < K; c++){
		for(uint r = 0; r < k*k*C; r++){
			reshaped_filters[r*K+c] = kernel[r+c*k*k*C];
		}
	}
	//Reshaping activations and collecting shards: Shards shape [out_rows*out_cols, k*k*C] when using SGEMM, otherwise [out_rows, out_cols, k*k*C]
	for(uint r = 0; r < out_rows; r++){
		for(uint c = 0; c < out_cols; c++){
			for(uint ch = 0; ch < C; ch++){
				for(uint u = 0; u < k; u++){
					for(uint v = 0; v < k; v++){
						temp0 = uneven ? r*stride + u : r*stride - padding + u;		//Check if physical padding is better than using conditionals in this loop?
						temp1 = uneven ? c*stride + v : c*stride - padding + v;
						if(temp0 < 0 || temp0 >= H || temp1 < 0 || temp1 >= W){
#if	USE_CBLAS
							shards[u*k+v + k*k*C*c + k*k*C*out_cols*r + ch*k*k] = 0;
#else
							shards[v*out_rows*out_cols+u*k*out_rows*out_cols+ch*k*k*out_rows*out_cols+(c+r*out_cols)] = 0;
#endif
						}else{
#if	USE_CBLAS
							shards[u*k+v + k*k*C*c + k*k*C*out_cols*r + ch*k*k] = data[ch*H*W+(temp0)*W+(temp1)];
#else
							shards[v*out_rows*out_cols+u*k*out_rows*out_cols+ch*k*k*out_rows*out_cols+(c+r*out_cols)] = data[ch*H*W+(temp0)*W+(temp1)];
#endif
						}
					}
				}	
			}
		}
	}
#if	USE_CBLAS	
//	clock_t begin = clock();
	_cblas_sgemm_stub(shards, reshaped_filters, _cblas_gemm_out, out_rows*out_cols, k*k*C, k*k*C, K);	
//	clock_t end = clock();
//	printf("GEMM took %f\r\n", ((double)(end - begin))/CLOCKS_PER_SEC);
	//cblas_sgemm only computes matrix matrix mul and not tesor tensor products, generated output is shaped [out_rows*out_cols, K], hence need to reshape output back to [out_rows, out_cols, K]
	for(uint r = 0; r < out_rows; r++){
		for(uint c = 0; c < out_cols; c++){
			for(uint ke = 0; ke < K; ke++){
				out[c+r*out_rows+ke*out_rows*out_cols] = _cblas_gemm_out[ke+c*K+r*out_cols*K];
			}
		}
	}
#else
	//Clear output mem locations
	memset(out, 0, out_rows*out_cols*K*sizeof(float));
	//Perform GEMM using loops
	for(uint r = 0; r < out_rows; r++){
		for(uint c = 0; c < out_cols; c++){
			for(uint kern = 0; kern < K; kern++){
				for(uint ch = 0; ch < k*k*C; ch++){	
					out[c+r*out_cols+kern*out_rows*out_cols] += shards[c+r*out_cols+ch*out_rows*out_cols] * reshaped_filters[kern+K*ch];
				}
			}
		}
	}
#endif
//	for(uint a = 0; a < out_rows*out_cols*K; a++){
//		printf("%f, ", _cblas_gemm_out[a]);
//	}
//	t_net_e = clock();
//	printf("Net time for Conv: %f\r\n", ((double)(t_net_e - t_net_s)/CLOCKS_PER_SEC));
	delete[] reshaped_filters;
	delete[] shards;
#if	USE_CBLAS
	delete[] _cblas_gemm_out;
#endif
}


/*
int main(void){
	uint fmap_size = 100000, k = 3, K = 24, C = 3, padding = 1, stride = 2;
	uint out_act_size = int((fmap_size-k+2*padding)/stride)+1;
	uint sgemm_mat0_dim0 = 5*5;
	uint sgemm_mat0_dim1 = k*k*C;
	uint sgemm_mat1_dim0 = k*k*C;
	uint sgemm_mat1_dim1 = K;
	uint sgemm_res_dim0 = sgemm_mat0_dim0;
	uint sgemm_res_dim1 = sgemm_mat1_dim1;

	float *in_act = new float[fmap_size*fmap_size*C];
	float *in_kern = new float[k*k*C*K];
	float *out_act = new float[out_act_size*out_act_size*K];
	float *sgemm_mat0 = new float[sgemm_mat0_dim0*sgemm_mat0_dim1];
	float *sgemm_mat1 = new float[sgemm_mat1_dim0*sgemm_mat1_dim1];
	float *sgemm_res = new float[sgemm_res_dim0*sgemm_res_dim1];
	for(uint i = 0; i < fmap_size*fmap_size*C; i++){
		in_act[i] = float(i)/float(fmap_size*fmap_size*C);
	}
	for(uint i = 0; i < k*k*C*K; i++){
		in_kern[i] = float(i)/float(k*k*C*K);
	}
	for(uint i = 0; i < sgemm_mat0_dim0*sgemm_mat0_dim1; i++){
		sgemm_mat0[i] = float(i)/float(sgemm_mat0_dim0*sgemm_mat0_dim1);
	}
	for(uint i = 0; i < sgemm_mat1_dim0*sgemm_mat1_dim1; i++){
		sgemm_mat1[i] = float(i)/float(sgemm_mat1_dim0*sgemm_mat1_dim1);
	}
	clock_t begin = clock();
	conv2d(out_act, in_kern, in_act, K, k, fmap_size, fmap_size, C, padding, stride);
//	_cblas_sgemm_stub(sgemm_mat0, sgemm_mat1, sgemm_res, sgemm_mat0_dim0, sgemm_mat0_dim1, sgemm_mat1_dim0, sgemm_mat1_dim1);
//	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, sgemm_mat0_dim0, sgemm_mat1_dim1, sgemm_mat0_dim1, 1.0, sgemm_mat0, sgemm_mat0_dim1, sgemm_mat1, sgemm_mat0_dim1, 0.0, sgemm_res, sgemm_res_dim1);
	clock_t end = clock();
	printf("Seconds %f\r\n", ((double)(end-begin))/CLOCKS_PER_SEC);
	
//	for(uint i = 0; i < out_act_size*out_act_size*K; i++){
//		printf("%f, ", out_act[i]);
//	}
//	for(uint i = 0; i < 2*2; i++){
//		printf("%f, ", sgemm_res[i]);
//	}
	printf("\r\n");
	delete[] in_act;
	delete[] in_kern;
	delete[] out_act;
	delete[] sgemm_mat0;
	delete[] sgemm_mat1;
	delete[] sgemm_res;
	return 0;
}
*/
