/*
 * correlation_loss_layer.cpp
 *
 *  Created on: 8 Apr, 2018
 *      Author: gathetaroot
 */


#include "cfloat"
#include "caffe/customLayers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void caffe_exp_loss(const int N,const int ch, const Dtype* x,const Dtype scaler,Dtype* sumY){
	Dtype loss = 0;
	sumY[0] = 0;
	for (int i = 0; i < N; i++){
		loss = 0;
		for (int j = 0; j < ch; j++){
			//loss += (fabs(x[i*ch + j]) + (exp(-1)*(exp(fabs(x[i*ch + j])) - 1.f)))*scaler;
			//loss+=(1.0-exp(-fabs(x[i*ch + j])))*scaler;
			loss+=((1.0+exp(-1.0))-(x[i*ch + j]+exp(-fabs(x[i*ch + j]))))*scaler;
		}
		sumY[0] += loss;
	}
	sumY[0] /= N;
}

template void caffe_exp_loss<int>(const int N,const int ch, const int* x,
		const int scaler,int* sumY);
template void caffe_exp_loss<float>(const int N,const int ch, const float* x,
		const float scaler,float* sumY);
template void caffe_exp_loss<double>(const int N,const int ch, const double* x,
		const double scaler,double* sumY);

template <typename Dtype>
void caffe_diff_exp_loss(const int N, const int ch, const Dtype*x, const Dtype scaler, Dtype* dx){
	Dtype loss = 0;
	float sign = 1;
	for (int i = 0; i < N; i++){
		for (int j = 0; j < ch; j++){
			sign = 1;
			if (x[i*ch + j] < 0)
				sign = -1;
			else if(x[i*ch + j]==0)
				sign=0;
				//dx[i*ch + j] = (x[i*ch + j] / fabs(x[i*ch + j]))*(1.f + (exp(-1)*(exp(fabs(x[i*ch + j])) - 1.f)))*scaler;
			//dx[i*ch + j] = sign*(1.f + (exp(-1)*(exp(fabs(x[i*ch + j])) - 1.f)))*scaler;
			//dx[i*ch + j]=sign*exp(-fabs(x[i*ch + j]))*scaler;
			dx[i*ch + j]=(1.0-sign*exp(-fabs(x[i*ch + j])))*scaler;
		}
	}
}
template void caffe_diff_exp_loss<int>(const int N, const int ch,
		const int*x, const int scaler, int* dx);
template void caffe_diff_exp_loss<double>(const int N, const int ch,
		const double*x, const double scaler, double* dx);
template void caffe_diff_exp_loss<float>(const int N, const int ch,
		const float*x, const float scaler, float* dx);

template<typename Dtype>
void ExpDecayLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>&bottom,
    	    				const vector<Blob<Dtype>*>&top){
	vector<int> in_shape=bottom[0]->shape();
	int dim1=1;
	if(in_shape.size()>1)
		dim1=in_shape[1];
	CHECK(in_shape.size()<=2&&dim1==1)<<"The dim of input must be scalers";
	_scaler=this->layer_param_.expdecayloss_param().scaler();
}

template<typename Dtype>
void ExpDecayLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>&bottom,
    	    				const vector<Blob<Dtype>*>&top){
	vector<int> new_shape(0);
	top[0]->Reshape(new_shape);
}

template<typename Dtype>
void ExpDecayLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>&bottom,
            				const vector<Blob<Dtype>*>&top){
	const Dtype* x=bottom[0]->cpu_data();
	Dtype* y=top[0]->mutable_cpu_data();
	int dim0=bottom[0]->shape()[0];
	caffe_exp_loss(dim0,1,x,_scaler,y);
}

template<typename Dtype>
void ExpDecayLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>&top,
            				const vector<bool>& propagation_down,
        					const vector<Blob<Dtype>*>&bottom){
	Dtype* x_diff=bottom[0]->mutable_cpu_diff();
	int dim0=bottom[0]->shape()[0];
	if(propagation_down[0]){
		const Dtype* x=bottom[0]->cpu_data();
		caffe_diff_exp_loss(dim0,1,x,_scaler,x_diff);
		caffe_scal(dim0,(Dtype)dim0,x_diff);
	}else{Dtype* x_diff=bottom[0]->mutable_cpu_diff();
		for(int i=0;i<dim0;i++){
			x_diff[i]=(Dtype)1;
		}
	}

}

#ifdef CPU_ONLY
STUB_GPU(ExpDecayLossLayer);
#endif

INSTANTIATE_CLASS(ExpDecayLossLayer);
REGISTER_LAYER_CLASS(ExpDecayLoss);

}

