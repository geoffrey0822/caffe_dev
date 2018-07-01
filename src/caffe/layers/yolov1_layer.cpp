/*
 * yolov1_layer.cpp
 *
 *  Created on: 29 Jun, 2018
 *      Author: gathetaroot
 */

#include "cfloat"
#include "caffe/layers/yolov1_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{

template <typename Dtype>
void caffe_yolo1(const int N,const Dtype*X,const Dtype*Gt,
		const int slide,const int nClass,const int nBox,
		const float scaleCoord,const float scaleNoObj,
		const float threshold,Dtype*Y){
	Dtype centerLoss,sizeLoss,objLoss, noObjLoss,clsLoss;
	Dtype tmp_centerLoss,tmp_sizeLoss,tmp_objLoss, tmp_noObjLoss,tmp_clsLoss;
	Dtype dx,dy,dw,dh,dstatus,dclass;
	// blob slide: slide*((1+4)*nBox+nClass)
	int blobSlide=slide*((1+4)*nBox+nClass);
	int boxSlide=(1+4)*nBox+nClass;
	int localN=N*nBox*slide;
	int classN=N*nClass*slide;

	bool hasObj=false;
	float largestConf=0;
	Y[0]=0;
	for(int i=0;i<N;i++){
		centerLoss=0;
		sizeLoss=0;
		noObjLoss=0;
		objLoss=0;
		clsLoss=0;
		tmp_centerLoss=0;
		tmp_sizeLoss=0;
		tmp_noObjLoss=0;
		tmp_objLoss=0;
		hasObj=false;
		largestConf=0;
		for(int j=0;j<slide;j++){
			for(int k=0;k<nBox;k++){
				dx=Gt[i*blobSlide+j*boxSlide+5*k]-X[i*blobSlide+j*boxSlide+5*k];
				dy=Gt[i*blobSlide+j*boxSlide+5*k+1]-X[i*blobSlide+j*boxSlide+5*k+1];
				dw=Gt[i*blobSlide+j*boxSlide+5*k+2]-X[i*blobSlide+j*boxSlide+5*k+2];
				dh=Gt[i*blobSlide+j*boxSlide+5*k+3]-X[i*blobSlide+j*boxSlide+5*k+3];
				dstatus=Gt[i*blobSlide+j*boxSlide+5*k+4]-X[i*blobSlide+j*boxSlide+5*k+4];

				if(X[i*blobSlide+j*boxSlide+5*k+4]>=largestConf){
					largestConf= X[i*blobSlide+j*boxSlide+5*k+4];
					tmp_centerLoss=(dx*dx)+(dy*dy);
					tmp_sizeLoss=(dw*dw)+(dh*dh);
					tmp_noObjLoss=(dstatus*dstatus);
					tmp_objLoss=tmp_noObjLoss;
				}

				//centerLoss+=(dx*dx)+(dy*dy);
				//sizeLoss+=(dw*dw)+(dh*dh);
				//noObjLoss+=(dstatus*dstatus);
				//objLoss=noObjLoss;
				if(X[i*blobSlide+j*boxSlide+5*k+4]>=threshold){
					hasObj=true;
				}

			}
			if(hasObj){
				for(int l=0;l<nClass;l++){
					dclass=Gt[i*blobSlide+j*boxSlide+5*nBox+l]-X[i*blobSlide+j*boxSlide+5*nBox+l];
					clsLoss+=dclass*dclass;
				}
			}
			centerLoss+=tmp_centerLoss;
			sizeLoss+=tmp_sizeLoss;
			noObjLoss+=tmp_noObjLoss;
			objLoss=noObjLoss;
		}
		Y[0]+=(scaleCoord*centerLoss+scaleCoord*sizeLoss+objLoss+scaleNoObj*noObjLoss)/(2*localN)+clsLoss/(2*classN);
	}
}

template void caffe_yolo1<double>(const int N,const double*X,const double*Gt,
		const int slide,const int nClass,const int nBox,
		const float scaleCoord,const float scaleNoObj,
		const float threshold,double*Y);
template void caffe_yolo1<float>(const int N,const float*X,const float*Gt,
		const int slide,const int nClass,const int nBox,
		const float scaleCoord,const float scaleNoObj,
		const float threshold,float*Y);

template <typename Dtype>
void caffe_dyolo1(const int N,const Dtype*X,const Dtype*Gt,
		const int slide,const int nClass,const int nBox,
		const float scaleCoord,const float scaleNoObj,
		const float threshold,Dtype*Y){
	Dtype dx,dy,dw,dh,dstatus,dclass;
	Dtype dldx,dldy,dldw,dldh,dldstatus,dldclass;
	// blob slide: slide*((1+4)*nBox+nClass)
	int blobSlide=slide*((1+4)*nBox+nClass);
	int boxSlide=(1+4)*nBox+nClass;

	float largestConf=0;
	bool hasObj=false;
	for(int i=0;i<N;i++){
		for(int j=0;j<slide;j++){
			hasObj=false;
			for(int k=0;k<nBox;k++){
				dx=Gt[i*blobSlide+j*boxSlide+5*k]-X[i*blobSlide+j*boxSlide+5*k];
				dy=Gt[i*blobSlide+j*boxSlide+5*k+1]-X[i*blobSlide+j*boxSlide+5*k+1];
				dw=Gt[i*blobSlide+j*boxSlide+5*k+2]-X[i*blobSlide+j*boxSlide+5*k+2];
				dh=Gt[i*blobSlide+j*boxSlide+5*k+3]-X[i*blobSlide+j*boxSlide+5*k+3];
				dstatus=Gt[i*blobSlide+j*boxSlide+5*k+4]-X[i*blobSlide+j*boxSlide+5*k+4];

				dldx=0;
				dldy=0;
				dldw=0;
				dldh=0;
				dldstatus=0;

				if(X[i*blobSlide+j*boxSlide+5*k+4]>=largestConf){
					largestConf=X[i*blobSlide+j*boxSlide+5*k+4];
					dldx=-dy*scaleCoord;
					dldy=dx*scaleCoord;
					dldw=-dh*scaleCoord;
					dldh=dw*scaleCoord;
					dldstatus=-(1+scaleNoObj)*dstatus;

				}

				if(X[i*blobSlide+j*boxSlide+5*k+4]>=threshold)
					hasObj=true;

				Y[i*blobSlide+j*boxSlide+j*boxSlide+5*k]=dldx;
				Y[i*blobSlide+j*boxSlide+j*boxSlide+5*k+1]=dldy;
				Y[i*blobSlide+j*boxSlide+j*boxSlide+5*k+2]=dldw;
				Y[i*blobSlide+j*boxSlide+j*boxSlide+5*k+3]=dldh;
				Y[i*blobSlide+j*boxSlide+j*boxSlide+5*k+4]=dldstatus;

			}
			for(int l=0;l<nClass;l++){
				dclass=Gt[i*blobSlide+j*boxSlide+5*nBox+l]-X[i*blobSlide+j*boxSlide+5*nBox+l];
				if(hasObj)
					dldclass=-X[i*blobSlide+j*boxSlide+5*nBox+l];
				else
					dldclass=0;
				Y[i*blobSlide+j*boxSlide+5*nBox+l]=dldclass;
			}
		}
	}
}


template void caffe_dyolo1<double>(const int N,const double*X,const double*Gt,
		const int slide,const int nClass,const int nBox,
		const float scaleCoord,const float scaleNoObj,
		const float threshold,double*Y);
template void caffe_dyolo1<float>(const int N,const float*X,const float*Gt,
		const int slide,const int nClass,const int nBox,
		const float scaleCoord,const float scaleNoObj,
		const float threshold,float*Y);

template<typename Dtype>
void YoloV1Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>&bottom,
		const vector<Blob<Dtype>*>&top){
	this->nClass=this->layer_param_.yolo_v1_param().class_num();
	this->nBox=this->layer_param_.yolo_v1_param().box_num();
	this->nGrid=this->layer_param_.yolo_v1_param().grid_size();
	this->threshold=this->layer_param_.yolo_v1_param().threshold();
	this->scaleNoObj=this->layer_param_.yolo_v1_param().gamma_noobj();
	this->scaleXY=this->layer_param_.yolo_v1_param().gamma_coord();

	CHECK(bottom.size()==2)<<"The number of input must be 2 for YOLO V1 Loss Layer.";

	vector<int> in_shape=bottom[0]->shape();
	vector<int> inGT_shape=bottom[1]->shape();

	CHECK_EQ(in_shape[1],inGT_shape[1]);

	CHECK_GE(this->nClass,2);
	CHECK_GE(this->nBox,1);
	CHECK_GE(this->nGrid,1);

}

template<typename Dtype>
void YoloV1Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>&bottom,
		const vector<Blob<Dtype>*>& top){
	int gt_input_size=this->nGrid*this->nGrid*(5*this->nBox+this->nClass);
	int input_size=bottom[0]->shape()[1];
	CHECK_EQ(input_size,gt_input_size);
}

template<typename Dtype>
void YoloV1Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
	const Dtype* x=bottom[0]->cpu_data();
	const Dtype* gt=bottom[1]->cpu_data();
	int nBatch=bottom[0]->shape()[0];
	int slide=this->nGrid*this->nGrid;
	Dtype* y=top[0]->mutable_cpu_data();

	caffe_yolo1(nBatch,x,gt,slide,this->nClass,this->nBox,
			this->scaleXY,this->scaleNoObj,this->threshold,y);
}

template<typename Dtype>
void YoloV1Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagation_down,const vector<Blob<Dtype>*>& bottom){
	const Dtype* x=bottom[0]->cpu_data();
	const Dtype* gt=bottom[1]->cpu_data();
	int nBatch=bottom[0]->shape()[0];
	int slide=this->nGrid*this->nGrid;
	Dtype* y=bottom[0]->mutable_cpu_diff();
	caffe_dyolo1(nBatch,x,gt,slide,this->nClass,this->nBox,
			this->scaleXY,this->scaleNoObj,this->threshold,y);
}

#ifdef CPU_ONLY
STUB_GPU(YoloV1Layer)
#endif

INSTANTIATE_CLASS(YoloV1Layer);
REGISTER_LAYER_CLASS(YoloV1);

}

