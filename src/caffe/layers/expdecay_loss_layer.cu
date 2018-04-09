/*
 * correlation_loss_layer.cu
 *
 *  Created on: 8 Apr, 2018
 *      Author: gathetaroot
 */

#include "cfloat"
#include "caffe/customLayers.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe{

template <typename Dtype>
__global__ void gpu_exp_loss_kernel(const int N, const int ch, const Dtype*x, const Dtype scaler, Dtype*y){
	y[0]=0;
	CUDA_KERNEL_LOOP(i, N){
		float loss = 0;
		for (int j = 0; j < ch; j++){
			//loss += (fabs(x[i*ch + j]) + (exp(-1.0)*(exp(fabs(x[i*ch + j])) - 1.0)))*scaler;
			loss+=(1.0-exp(-fabs(x[i*ch + j])))*scaler;
		}
		y[0] += loss/(Dtype)N;
	}
}

template <typename Dtype>
__global__ void gpu_diff_exp_loss_kernel(const int N, const int ch, const Dtype*x, const Dtype scaler, Dtype*dx){
	CUDA_KERNEL_LOOP(i, N){
		double sign = 1.0;
		for (int i = 0; i < N; i++){
			for (int j = 0; j < ch; j++){
				sign = 1.0;
				if (x[i*ch + j] < 0)
					sign = -1;
				else if(x[i*ch + j]==0)
					sign=0;
				//dx[i*ch + j] = (x[i*ch + j] / fabs(x[i*ch + j]))*(1.f + (exp(-1.f)*(exp(fabs(x[i*ch + j])) - 1.f)))*scaler;
				//dx[i*ch + j] = sign*(1.0 + (exp(-1.0)*(exp(fabs(x[i*ch + j])) - 1.0)))*scaler;
				dx[i*ch + j]=sign*exp(-fabs(x[i*ch + j]))*scaler;
			}
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
void gpu_exp_loss(const int N, const int ch, const Dtype* x, const Dtype scaler, Dtype* sumY){
	gpu_exp_loss_kernel<< <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> >(N,ch,x,scaler,sumY);
}

//template void gpu_exp_loss<int>(const int N, const int ch, const int* x, const int scaler, int* sumY);
template void gpu_exp_loss<double>(const int N, const int ch, const double* x, const double scaler, double* sumY);
template void gpu_exp_loss<float>(const int N, const int ch, const float* x, const float scaler, float* sumY);

template <typename Dtype>
void gpu_diff_exp_loss(const int N, const int ch, const Dtype* x, const Dtype scaler, Dtype* dx){
	gpu_diff_exp_loss_kernel<< <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> >(N,ch,x,scaler,dx);
}

//template void gpu_diff_exp_loss<int>(const int N, const int ch, const int* x, const int scaler, int* dx);
template void gpu_diff_exp_loss<double>(const int N, const int ch, const double* x, const double scaler, double* dx);
template void gpu_diff_exp_loss<float>(const int N, const int ch, const float* x, const float scaler, float* dx);

template <typename Dtype>
void gpu_set_value(const int N,const Dtype val,Dtype* y){
	gpu_set_value_kernel<< <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> >(N,val,y);
}
template void gpu_set_value<int>(const int N,const int val,int *y);
template void gpu_set_value<double>(const int N,const double val,double *y);
template void gpu_set_value<float>(const int N,const float val,float *y);

template <typename Dtype>
void ExpDecayLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>&bottom,
            				const vector<Blob<Dtype>*>&top){
	const Dtype* x=bottom[0]->gpu_data();
	Dtype* y=top[0]->mutable_gpu_data();
	int dim0=bottom[0]->shape()[0];
	gpu_exp_loss(dim0,1,x,(Dtype)_scaler,y);
}

template <typename Dtype>
void ExpDecayLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>&top,
            		const vector<bool>& propagation_down,
            		const vector<Blob<Dtype>*>&bottom){

	Dtype* x_diff=bottom[0]->mutable_gpu_diff();
	int dim0=bottom[0]->shape()[0];
	if(propagation_down[0]){
		const Dtype* x=bottom[0]->gpu_data();
		gpu_diff_exp_loss(dim0,1,x,(Dtype)_scaler,x_diff);
	}else{
		gpu_set_value(dim0,(Dtype)1,x_diff);
	}
}


INSTANTIATE_LAYER_GPU_FUNCS(ExpDecayLossLayer);

}

