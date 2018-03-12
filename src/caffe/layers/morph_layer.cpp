/*
 *
 * morph_layer.cpp
 *
 *  Created on: 9 Mar 2018
 *      Author: Geoffrey Poon
 */

#include "cfloat"
#include "caffe/customLayers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void caffe_linear_proj(const int Nm,
		const Dtype* X,const unsigned int* Z,Dtype* Y){
	for(int i=0;i<Nm;i++){
		Y[i]=X[Z[i]];
	}
}

template void caffe_linear_proj<int>(const int Nm,const int* X,const unsigned int* Z,int* Y);
template void caffe_linear_proj<double>(const int Nm,const double* X,const unsigned int* Z,double* Y);
template void caffe_linear_proj<float>(const int Nm,const float* X,const unsigned int* Z,float* Y);

template <typename Dtype>
void MorphLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top){
	CHECK(bottom.size()==2)<<"Only support 2 inputs";
	CHECK(top.size()==2)<<"Only support 2 outputs";
	vector<int> input1_shape=bottom[0]->shape();
	vector<int> input2_shape=bottom[1]->shape();
	printf("[2]=%d<>%d,[3]=%d<>%d\n",input1_shape[2],input2_shape[2],input1_shape[3],input2_shape[3]);
	CHECK((input1_shape[2]==input2_shape[2]&&
					input1_shape[3]==input2_shape[3]))<<"Dimension of axis 2 and 3 must be the same";
	num_=input1_shape[0];
	ch_size_=input1_shape[1];
	w_=input1_shape[3];
	h_=input1_shape[2];
	if(w_>0 && h_>0){
		printf("Setup for Morph Layer Reshape(%d,%d,%d,%d)\n",num_,ch_size_,w_,h_);
		map_.Reshape(num_,ch_size_,h_,w_);
		unsigned int* data=map_.mutable_cpu_data();
		int scaler=input2_shape[1];
		size_t index;
		for(int n=0;n<num_;n++){
			for(int i=0;i<ch_size_;i++){
				for(int j=0;j<h_;j++){
					for(int k=0;k<w_;k++){
						index=n*ch_size_*h_*w_+i*h_*w_+j*w_+k;
						data[index]=(int)floor(((float)(i*h_*w_+j*w_+k)*scaler)/ch_size_);
					}
				}
			}
		}
	}else{
		vector<int>new_shape(2);
		new_shape.push_back(num_);
		new_shape.push_back(ch_size_);
		h_=1;
		w_=1;
		map_.Reshape(num_,ch_size_,h_,w_);
		printf("Setup for Morph Layer Reshape(%d,%d,%d,%d)\n",num_,ch_size_,w_,h_);
		unsigned int* data=map_.mutable_cpu_data();
		int scaler=input2_shape[1];
		size_t index;
		for(int n=0;n<num_;n++){
			for(int i=0;i<ch_size_;i++){
				for(int j=0;j<h_;j++){
					for(int k=0;k<w_;k++){
						index=n*ch_size_*h_*w_+i*h_*w_+j*w_+k;
						data[index]=(int)floor(((float)(i*h_*w_+j*w_+k)*scaler)/ch_size_);
					}
				}
			}
		}
	}
	printf("Setup Finished\n");
}

template <typename Dtype>
void MorphLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top){

	vector<int> top_shape=bottom[0]->shape();
	top[0]->Reshape(top_shape);
	top[1]->Reshape(top_shape);
}

template <typename Dtype>
void MorphLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top){
	const Dtype* bottom1_data=bottom[0]->cpu_data();
	const Dtype* bottom2_data=bottom[1]->cpu_data();
	Dtype* top1_data=top[0]->mutable_cpu_data();
	const unsigned int* map_data=map_.cpu_data();
	Dtype* top2_data=top[1]->mutable_cpu_data();
	size_t count=bottom[0]->count();

	caffe_copy<Dtype>(count,bottom1_data,top1_data);
	caffe_linear_proj<Dtype>(count,bottom2_data,map_data,top2_data);
}

template <typename Dtype>
void MorphLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagation_down, const vector<Blob<Dtype> *> &bottom){
	Dtype* bottom1_diff=bottom[0]->mutable_cpu_diff();
	Dtype* bottom2_diff=bottom[1]->mutable_cpu_diff();
	const Dtype* top1_diff=top[0]->cpu_diff();

	caffe_copy<Dtype>(bottom[0]->count(),top1_diff,bottom1_diff);
	caffe_set<Dtype>(bottom[1]->count(),Dtype(0),bottom2_diff);
}


#ifdef CPU_ONLY
STUB_GPU(MorphLayer);
#endif

INSTANTIATE_CLASS(MorphLayer);
REGISTER_LAYER_CLASS(Morph);
} // namespace caffe
