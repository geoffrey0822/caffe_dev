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
#include <sstream>
#include <string>
#include <fstream>

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
	CHECK(top.size()==1)<<"Only support 1 output";
	vector<int> input1_shape=bottom[0]->shape();
	vector<int> input2_shape=bottom[1]->shape();
	int h2=1;
	int w2=1;
	w_=1;
	h_=1;
	if(input1_shape.size()>2){
		w_=input1_shape[3];
		h_=input1_shape[2];
	}
	if(input2_shape.size()>2){
		h2=input2_shape[2];
		w2=input2_shape[3];
	}
	printf("[2]=%d<>%d,[3]=%d<>%d\n",h_,h2,w_,w2);
	CHECK((h_==h2&&w_==w2))<<"Dimension of axis 2 and 3 must be the same";
	num_=input1_shape[0];
	ch_size_=input1_shape[1];
	int ch_size2=input2_shape[1];
	int newI=0;
	std::string index_file=this->layer_param_.morph_param().index_map();
	if(!index_file.empty()){
		CHECK(w_<=1&&h_<=1)<<"Morph layer only support NxCx1x1 arch";
		w_=1;
		h_=1;
		CHECK(boost::filesystem::exists(index_file))<<"Index map does not existed!";
		std::ifstream infile(index_file.c_str());
		std::string line;
		std::string value_str;
		int w=0,h=0,ch=0,max_w,max_h,i,j,k;
		bool newline=false;
		bool firstInit=false;
		while(std::getline(infile,line)){
			if(line.empty()){
				ch++;
				if(!firstInit){
					max_w=w;
					max_h=h;
					firstInit=true;
				}
				h=0;
				newline=true;
				continue;
			}
			w=0;
			std::istringstream ss(line);
			newline=false;
			while(std::getline(ss,value_str,',')){
				w++;
			}
			h++;
		}
		if(!newline)
			ch++;
		infile.close();
		//h+=1;
		//h+=1;w+=1;ch+=1;
		char message[255];
		memset(message,0,255);
		sprintf(message,"Invalid Dimension of index map!Index Map Dim:<%d,%d,%d> E<%d,%d,%d>",ch,h,w,ch_size_,h_,w_);
		CHECK(max_w==w_&&max_h==h_&&ch_size_==ch)<<message;
		h_=max_h;
		w_=max_w;
		ch_size_=ch;
		map_.Reshape(num_,ch_size_,h_,w_);
		unsigned int* data=map_.mutable_cpu_data();
		i=0;
		j=0;
		k=0;
		int idx=0;
		infile.open(index_file.c_str(),std::ifstream::in);
		while(std::getline(infile,line)){
			j=0;
			std::istringstream ss(line);
			if(line.empty()){
				k++;
				i=0;
				continue;
			}
			while(std::getline(ss,value_str,',')){
				for(int n=0;n<num_;n++){
					idx=atoi(value_str.c_str());
					//data[ch_size_*(w_*(n*h_+i)+j)+k]=n*ch_size2*h_*w_+idx*h_*w_+j*w_+k;
					data[ch_size_*(w_*(n*h_+i)+j)+k]=n*ch_size2*h_*w_+idx*h_*w_+i*w_+j;
				}
				j++;
			}
			i++;
		}
		infile.close();

	}else{
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
							newI=(int)floor((float)(i*scaler)/ch_size_);
							data[index]=n*ch_size2*h_*w_+newI*h_*w_+j*w_+k;
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
							newI=(int)floor((float)(i*scaler)/ch_size_);
							data[index]=n*ch_size2*h_*w_+newI*h_*w_+j*w_+k;
						}
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
	//top[1]->Reshape(top_shape);
}

template <typename Dtype>
void MorphLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top){
	const Dtype* bottom1_data=bottom[0]->cpu_data();
	const Dtype* bottom2_data=bottom[1]->cpu_data();
	//Dtype* top1_data=top[0]->mutable_cpu_data();
	const unsigned int* map_data=map_.cpu_data();
	Dtype* top2_data=top[0]->mutable_cpu_data();
	size_t count=bottom[0]->count();

	//caffe_copy<Dtype>(count,bottom1_data,top1_data);
	caffe_linear_proj<Dtype>(count,bottom2_data,map_data,top2_data);
}

template <typename Dtype>
void MorphLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagation_down, const vector<Blob<Dtype> *> &bottom){

	if(!propagation_down.empty()&&propagation_down[0]){
		Dtype* bottom1_diff=bottom[0]->mutable_cpu_diff();
		Dtype* bottom2_diff=bottom[1]->mutable_cpu_diff();
		const Dtype* top1_diff=top[0]->cpu_diff();

		//caffe_copy<Dtype>(bottom[0]->count(),top1_diff,bottom1_diff);
		caffe_set<Dtype>(bottom[1]->count(),Dtype(0),bottom2_diff);
	}
}


#ifdef CPU_ONLY
STUB_GPU(MorphLayer);
#endif

INSTANTIATE_CLASS(MorphLayer);
REGISTER_LAYER_CLASS(Morph);
} // namespace caffe
