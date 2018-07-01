/*
 * yolov1_layer.cu
 *
 *  Created on: 29 Jun, 2018
 *      Author: gathetaroot
 */

#include "cfloat"
#include "caffe/layers/yolov1_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{

template <typename Dtype>
__global__ void gpu_yolov1_loss_kernel(const int N,
		const Dtype*X,const Dtype* Gt,
		const int slide,const int nClass,const int nBox,
		const float scaleCoord,const float scaleNoObj,
		const float threshold,Dtype*Y){
	CUDA_KERNEL_LOOP(i,N){
		if(i==0)
			Y[0]=0;
		Dtype centerLoss,sizeLoss,objLoss,noObjLoss,clsLoss;
		Dtype tmp_centerLoss,tmp_sizeLoss,tmp_objLoss,tmp_noObjLoss,tmp_clsLoss;
		Dtype dx,dy,dw,dh,dstatus,dclass;
		float largestConf=0;
		int blobSlide=slide*(5*nBox+nClass);
		int boxSlide=5*nBox+nClass;
		int localN=N*nBox*slide;
		int classN=N*nClass*slide;
		bool hasObj=false;
		centerLoss=0;
		sizeLoss=0;
		noObjLoss=0;
		objLoss=0;
		clsLoss=0;
		for(int j=0;j<slide;j++){
			tmp_centerLoss=0;
			tmp_sizeLoss=0;
			tmp_noObjLoss=0;
			tmp_objLoss=0;
			for(int k=0;k<nBox;k++){
				dx=Gt[i*blobSlide+j*boxSlide+5*k]-X[i*blobSlide+j*boxSlide+5*k];
				dy=Gt[i*blobSlide+j*boxSlide+5*k+1]-X[i*blobSlide+j*boxSlide+5*k+1];
				dw=Gt[i*blobSlide+j*boxSlide+5*k+2]-X[i*blobSlide+j*boxSlide+5*k+2];
				dh=Gt[i*blobSlide+j*boxSlide+5*k+3]-X[i*blobSlide+j*boxSlide+5*k+3];
				dstatus=Gt[i*blobSlide+j*boxSlide+5*k+4]-X[i*blobSlide+j*boxSlide+5*k+4];

				if(X[i*blobSlide+j*boxSlide+5*k+4]>=largestConf){
					tmp_centerLoss=(dx*dx)+(dy*dy);
					tmp_sizeLoss=(dw*dw)+(dh*dh);
					tmp_noObjLoss=(dstatus*dstatus);
					tmp_objLoss=noObjLoss;
				}

				//centerLoss+=(dx*dx)+(dy*dy);
				//sizeLoss+=(dw*dw)+(dh*dh);
				//noObjLoss+=(dstatus*dstatus);
				//objLoss=noObjLoss;
				if(X[i*blobSlide+j*boxSlide+5*k+4]>=threshold)
					hasObj=true;
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
			objLoss+=tmp_objLoss;
		}
		Y[0]+=(scaleCoord*centerLoss+scaleCoord*sizeLoss+objLoss+scaleNoObj*noObjLoss)/(2*localN)+clsLoss/(2*classN);
	}
}

template <typename Dtype>
__global__ void gpu_dyolov1_loss_kernel(const int N,
		const Dtype*X,const Dtype* Gt,
		const int slide,const int nClass,const int nBox,
		const float scaleCoord,const float scaleNoObj,
		const float threshold,Dtype*Y){
	CUDA_KERNEL_LOOP(i,N){
		Dtype dx,dy,dw,dh,dstatus,dclass;
		Dtype dldx,dldy,dldw,dldh,dldstatus,dldclass;
		int blobSlide=slide*(5*nBox+nClass);
		int boxSlide=5*nBox+nClass;
		bool hasObj=false;
		for(int j=0;j<slide;j++){
			for(int k=0;k<nBox;k++){
				dx=Gt[i*blobSlide+j*boxSlide+5*k]-X[i*blobSlide+j*boxSlide+5*k];
				dy=Gt[i*blobSlide+j*boxSlide+5*k+1]-X[i*blobSlide+j*boxSlide+5*k+1];
				dw=Gt[i*blobSlide+j*boxSlide+5*k+2]-X[i*blobSlide+j*boxSlide+5*k+2];
				dh=Gt[i*blobSlide+j*boxSlide+5*k+3]-X[i*blobSlide+j*boxSlide+5*k+3];
				dstatus=Gt[i*blobSlide+j*boxSlide+5*k+4]-X[i*blobSlide+j*boxSlide+5*k+4];

				dldx=-dy*scaleCoord;
				dldy=dx*scaleCoord;
				dldw=-dh*scaleCoord;
				dldh=dw*scaleCoord;
				dldstatus=-(1+scaleNoObj)*dstatus;
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

template <typename Dtype>
void caffe_gpu_yolo1(const int N,const Dtype*X,const Dtype*Gt,
		const int slide,const int nClass,const int nBox,
		const float scaleCoord,const float scaleNoObj,
		const float threshold,Dtype*Y){
	gpu_yolov1_loss_kernel<< <CAFFE_GET_BLOCKS(N),CAFFE_CUDA_NUM_THREADS>> >(N,
			X,Gt,slide,nClass,
			nBox,scaleCoord,scaleNoObj,threshold,Y);
}

template void caffe_gpu_yolo1<float>(const int N,const float*X,const float*Gt,
		const int slide,const int nClass,const int nBox,
		const float scaleCoord,const float scaleNoObj,
		const float threshold,float*Y);

template void caffe_gpu_yolo1<double>(const int N,const double*X,const double*Gt,
		const int slide,const int nClass,const int nBox,
		const float scaleCoord,const float scaleNoObj,
		const float threshold,double*Y);

template <typename Dtype>
void caffe_gpu_dyolo1(const int N,const Dtype*X,const Dtype*Gt,
		const int slide,const int nClass,const int nBox,
		const float scaleCoord,const float scaleNoObj,
		const float threshold,Dtype*Y){
	gpu_dyolov1_loss_kernel<< <CAFFE_GET_BLOCKS(N),CAFFE_CUDA_NUM_THREADS>> >(N,
			X,Gt,slide,nClass,nBox,scaleCoord,
			scaleNoObj,threshold,Y);
}

template void caffe_gpu_dyolo1<float>(const int N,const float*X,const float*Gt,
		const int slide,const int nClass,const int nBox,
		const float scaleCoord,const float scaleNoObj,
		const float threshold,float*Y);

template void caffe_gpu_dyolo1<double>(const int N,const double*X,const double*Gt,
		const int slide,const int nClass,const int nBox,
		const float scaleCoord,const float scaleNoObj,
		const float threshold,double*Y);

template<typename Dtype>
void YoloV1Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
	const Dtype* x=bottom[0]->gpu_data();
	const Dtype* gt=bottom[1]->gpu_data();
	int nBatch=bottom[0]->shape()[0];
	int slide=this->nGrid*this->nGrid;
	Dtype* y=top[0]->mutable_gpu_data();
	caffe_yolo1(nBatch,x,gt,slide,this->nClass,this->nBox,
				this->scaleXY,this->scaleNoObj,this->threshold,y);
}

template<typename Dtype>
void YoloV1Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagation_down,const vector<Blob<Dtype>*>& bottom){
	const Dtype* x=bottom[0]->gpu_data();
	const Dtype* gt=bottom[1]->gpu_data();
	int nBatch=bottom[0]->shape()[0];
	int slide=this->nGrid*this->nGrid;
	Dtype* y=bottom[0]->mutable_gpu_diff();
	caffe_dyolo1(nBatch,x,gt,slide,this->nClass,this->nBox,
			this->scaleXY,this->scaleNoObj,this->threshold,y);
}

INSTANTIATE_LAYER_GPU_FUNCS(YoloV1Layer);

}


