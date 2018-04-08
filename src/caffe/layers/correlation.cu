/*
 * correlation.cu
 *
 *  Created on: 3 Apr, 2018
 *      Author: gathetaroot
 */

#include <vector>

#include "caffe/customLayers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void cov(const int N,const int dim, const Dtype* x, const Dtype* x_mean,
	const Dtype*y, const Dtype* y_mean, Dtype *z) {

	CUDA_KERNEL_LOOP(i, N) {
		float sum = 0;
		for (int j = 0; j < dim; j++)
			sum += (x[i*dim + j] - x_mean[i])*(y[i*dim + j] - y_mean[i]);
		z[i] = sum;
	}
}

template <typename Dtype>
__global__ void cdiv_kernel(const int N, const Dtype* x, Dtype divider,Dtype* y){
	CUDA_KERNEL_LOOP(i, N){
		y[i] =x[i]/ divider;
	}
}

template <typename Dtype>
__global__ void exp_decay(const int N,const Dtype* x, Dtype* y){
	CUDA_KERNEL_LOOP(i, N){
		y[i] = exp(-1 * fabs(x[i]));
	}
}

template <typename Dtype>
__global__ void norm_correl(const int N,const Dtype *mag, const Dtype *g2_x, const Dtype* g2_y,
		Dtype* z){
	CUDA_KERNEL_LOOP(i, N){
		z[i] = mag[i] / sqrt((g2_x[i] * g2_y[i]));
	}
}

template <typename Dtype>
__global__ void cproduct_kernel(const int N, const Dtype* x, const Dtype* y, Dtype* z){
	CUDA_KERNEL_LOOP(i, N){
		z[i] = x[i] * y[i];
	}
}

template <typename Dtype>
__global__ void csqrt_kernel(const int N, const Dtype*x, Dtype*z){
	CUDA_KERNEL_LOOP(i, N){
		z[i] = sqrt(x[i]);
	}
}

template <typename Dtype>
__global__ void G2_kernel(const int N, const Dtype* x, const Dtype* y, Dtype* z){
	CUDA_KERNEL_LOOP(i, N){
		z[i] = sqrt((x[i] * y[i]));
	}
}

template <typename Dtype>
__global__ void cscale_kernel(const int N, const Dtype* x, const Dtype scaler, Dtype* z){
	CUDA_KERNEL_LOOP(i, N){
		z[i] = x[i] * scaler;
	}
}

template <typename Dtype>
__global__ void correl_gradient(const int N, const int dim, const Dtype* x2,const Dtype* x2_mean,const Dtype* g1,
	const Dtype* g2,const Dtype*x1Term, const Dtype* X2, Dtype* z){
	CUDA_KERNEL_LOOP(i, N){
		for (int j = 0; j < dim; j++){
			z[i*dim + j] = (dim*(x2[i*dim + j] - x2_mean[i]) / (g2[i])) - (( x1Term[i] * X2[i] * g1[i]) / (g2[i] * g2[i] * g2[i]));
		}
	}
}

template <typename Dtype>
__global__ void vec_sum(const int N,const int dim, const Dtype*x, Dtype*y) {

	CUDA_KERNEL_LOOP(i, N) {
		float sum = 0;
		for (int j = 0; j < dim; j++){
			sum += x[i*dim + j];
		}
		y[i] = sum;
	}
}

template <typename Dtype>
__global__ void weighted_kernel(const int N, const int dim, const Dtype* x,const Dtype *w, Dtype *y){
	CUDA_KERNEL_LOOP(i, N){
		for (int j = 0; j < dim; j++){
			y[i*dim + j] = w[i] * x[i*dim + j];
		}
	}
}

template <typename Dtype>
__global__ void substract_mean(const int N, const int dim, const Dtype* x,const Dtype* y, Dtype* z){
	CUDA_KERNEL_LOOP(i, dim){
		for (int j = 0; j < dim; j++){
			z[i*dim + j] = x[i*dim + j] - y[i];
		}
	}
}

template <typename Dtype>
__global__ void gpu_set_value_kernel(const int N,const Dtype value,Dtype* y){
	CUDA_KERNEL_LOOP(i,N){
		y[i]=value;
	}
}

template <typename Dtype>
void weighted(const int N,const int dim,const Dtype* x,const Dtype *w, Dtype *y){
	weighted_kernel<< <CAFFE_GET_BLOCKS(1), CAFFE_CUDA_NUM_THREADS >> >(N,dim,x,w,y);
}

template void weighted<double>(const int N,const int dim,const double* x,const double *w, double *y);
template void weighted<float>(const int N,const int dim,const float* x,const float *w, float *y);


template <typename Dtype>
void caffe_gpu_pearson_correlation(const int n, const int k,
		const Dtype* x,const Dtype* y,Dtype* z,
		Dtype* x_mean,Dtype* y_mean,Dtype* g2_a,Dtype* g2_b,Dtype* g1){
	vec_sum << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, k, x, x_mean);
	cdiv_kernel << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, x_mean, (Dtype)k, x_mean);
	vec_sum << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, k, y, y_mean);
	cdiv_kernel << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, y_mean, (Dtype)k, y_mean);

	cov << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, k, x, x_mean, y, y_mean, g1);
	cov << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, k, x, x_mean, x, x_mean, g2_a);
	cov << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, k, y, y_mean, y, y_mean, g2_b);

	norm_correl << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, g1, g2_a, g2_b, z);
}

template void caffe_gpu_pearson_correlation<double>(const int N, const int k,
		const double* x,const double* y,double* z,
		double* x_mean,double* y_mean,double* g2_a,double* g2_b,double* g1);
template void caffe_gpu_pearson_correlation<float>(const int N, const int k,
		const float* x,const float* y,float* z,
		float* x_mean,float* y_mean,float* g2_a,float* g2_b,float* g1);


template <typename Dtype>
void caffe_gpu_diff_pearson_correlation(const int n, const int k,
		const Dtype* x, const Dtype* y, Dtype* dx,Dtype *dy,
		Dtype* X,Dtype* Y,
				Dtype* x_mean,Dtype* y_mean,Dtype* g2,Dtype* g1,
				Dtype* xTerm,Dtype *yTerm){
	vec_sum << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, k, x, x_mean);
	cdiv_kernel << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, x_mean, (Dtype)k, x_mean);
	vec_sum << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, k, y, y_mean);
	cdiv_kernel << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, y_mean, (Dtype)k, y_mean);

	substract_mean << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, k, x, x_mean, xTerm);
	vec_sum << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, k, xTerm, xTerm);
	substract_mean << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, k, y, y_mean, yTerm);
	vec_sum << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, k, yTerm, yTerm);

	cov << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, k, x, x_mean, y, y_mean, g1);
	cov << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, k, x, x_mean, x, x_mean, X);
	cov << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, k, y, y_mean, y, y_mean, Y);

	G2_kernel << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, X, Y, g2);

	correl_gradient << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, k, y, y_mean, g1, g2, xTerm, Y, dx);
	correl_gradient << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >(n, k, x, x_mean, g1, g2, yTerm, X, dy);

}

template void caffe_gpu_diff_pearson_correlation<float>(const int N, const int k,
		const float* x, const float* y, float* dx,float *dy,
		float* X,float* Y,
		float* x_mean,float* y_mean,float* g2,float* g1,
		float* xTerm,float *yTerm);
template void caffe_gpu_diff_pearson_correlation<double>(const int N, const int k,
		const double* x, const double* y, double* dx,double *dy,
		double* X,double* Y,
		double* x_mean,double* y_mean,double* g2,double* g1,
		double* xTerm,double *yTerm);

template <typename Dtype>
void gpu_set_value(const int N,const Dtype val,Dtype* y){
	gpu_set_value_kernel<< <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> >(N,val,y);
}
template void gpu_set_value<int>(const int N,const int val,int *y);
template void gpu_set_value<double>(const int N,const double val,double *y);
template void gpu_set_value<float>(const int N,const float val,float *y);


template <typename Dtype>
void CorrelationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>&bottom,
                             const vector<Blob<Dtype>*>&top){
	int dim0=bottom[0]->shape()[0];
	int dim1=bottom[0]->shape()[1];
	const Dtype* x_data=bottom[0]->gpu_data();
	const Dtype* y_data=bottom[1]->gpu_data();
	Dtype* z=top[0]->mutable_gpu_data();
	caffe_gpu_pearson_correlation(dim0,dim1,x_data,y_data,z,
			x_mean_.mutable_gpu_data(),y_mean_.mutable_gpu_data(),
			X_.mutable_gpu_data(),Y_.mutable_gpu_data(),
			g1_.mutable_gpu_data());

}


template <typename Dtype>
void CorrelationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>&top,
                               const vector<bool>& propagation_down,
                              const vector<Blob<Dtype>*>&bottom){

	int dim0=bottom[0]->shape()[0];
	int dim1=bottom[0]->shape()[1];
	if(!propagation_down.empty()&&propagation_down[0]){
		const Dtype* x=bottom[0]->gpu_data();
		const Dtype* y=bottom[1]->gpu_data();
		Dtype* x_diff=bottom[0]->mutable_gpu_diff();
		Dtype* y_diff=bottom[1]->mutable_gpu_diff();
		const Dtype* g_diff=top[0]->gpu_diff();

		if(propagation_down[0]){
			caffe_gpu_diff_pearson_correlation(dim0,dim1,x,y,x_diff,
					y_diff,X_.mutable_gpu_data(),Y_.mutable_gpu_data(),
					x_mean_.mutable_gpu_data(),
					y_mean_.mutable_gpu_data(),g2_.mutable_gpu_data(),
					g1_.mutable_gpu_data(),
					xTerm_.mutable_gpu_data(),
					yTerm_.mutable_gpu_data());

			const Dtype* top_diff=top[0]->gpu_data();
			weighted(dim0,dim1,x_diff,top_diff,x_diff);
			weighted(dim0,dim1,y_diff,top_diff,y_diff);
		}else{
			gpu_set_value(dim0,(Dtype)1,x_diff);
			gpu_set_value(dim0,(Dtype)1,y_diff);
		}

		//caffe_copy(bottom[0]->count(),top1_diff,bottom1_diff);

		//caffe_gpu_set(bottom[1]->count(),Dtype(0),bottom2_diff);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(CorrelationLayer);
}


