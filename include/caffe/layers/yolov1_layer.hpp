#ifndef YOLO_V1_LAYER_HPP
#define YOLO_V1_LAYER_HPP

#include <vector>
#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

template <typename Dtype>
void caffe_yolo1(const int N,const Dtype*X,const Dtype*Gt,
		const int slide,const int nClass,const int nBox,
		const float scaleCoord,const float scaleNoObj,
		const float threshold,Dtype*Y);

template <typename Dtype>
void caffe_gpu_yolo1(const int N,const Dtype*X,const Dtype*Gt,
		const int slide,const int nClass,const int nBox,
		const float scaleCoord,const float scaleNoObj,
		const float threshold,Dtype*Y);

template <typename Dtype>
void caffe_dyolo1(const int N,const Dtype*X,const Dtype*Gt,
		const int slide,const int nClass,const int nBox,
		const float scaleCoord,const float scaleNoObj,
		const float threshold,Dtype*Y);

template <typename Dtype>
void caffe_gpu_dyolo1(const int N,const Dtype*X,const Dtype*Gt,
		const int slide,const int nClass,const int nBox,
		const float scaleCoord,const float scaleNoObj,
		const float threshold,Dtype*Y);

template <typename Dtype>
class YoloV1Layer: public LossLayer<Dtype>{
public:
	explicit YoloV1Layer(const LayerParameter& param):
	LossLayer<Dtype>(param){}
	//Layer<Dtype>(param){}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>&bottom,
			const vector<Blob<Dtype>*>&top);

	virtual void Reshape(const vector<Blob<Dtype>*>&bottom,
			const vector<Blob<Dtype>*>&top);

	virtual inline const char* type()const{return "YoloV1Loss";}

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>&bottom,
			const vector<Blob<Dtype>*>&top);

	virtual void Forward_gpu(const vector<Blob<Dtype>*>&bottom,
				const vector<Blob<Dtype>*>&top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>&top,
			const vector<bool>& propagation_down,
			const vector<Blob<Dtype>*>&bottom);

	virtual void Backward_gpu(const vector<Blob<Dtype>*>&top,
			const vector<bool>& propagation_down,
				const vector<Blob<Dtype>*>&bottom);

	int nClass;
	int nGrid;
	int nBox;
	float threshold;
	float scaleXY;
	float scaleNoObj;
};

} //namespace caffe

#endif // YOLO_V1_LAYER_HPP
