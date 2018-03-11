#ifndef CAFFE_CUSTOM_LAYERS_HPP_
#define CAFFE_CUSTOM_LAYERS_HPP_

#include <vector>
#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

// Addition Mathematical Functions
template <typename Dtype>
void caffe_linear_proj(const int Nm,
		const Dtype* X,const unsigned int* Z,Dtype* Y);

template <typename Dtype>
void caffe_gpu_linear_proj(const int Nm,
		const Dtype* X,const unsigned int* Z,Dtype* Y);


//

    template <typename Dtype>
    class MorphLayer: public Layer<Dtype>{
        public:
    		explicit MorphLayer(const LayerParameter& param):
			Layer<Dtype>(param){}

    		virtual inline const char* type() const {return "Morph";}

    		virtual inline int MinBottomBlobs()const{return 2;}
    		virtual inline int MaxBottomBlobs()const{return 2;}
    		virtual inline int MinTopBlobs()const{return 2;}
    		virtual inline int MaxTopBlobs()const{return 2;}
    		virtual inline int ExactNumBottomBlobs() const { return 2; }
    		virtual inline int ExactNumTopBlobs() const { return 2; }

    		virtual void LayerSetUp(const vector<Blob<Dtype>*>&bottom,
    				const vector<Blob<Dtype>*>&top);

    		virtual void Reshape(const vector<Blob<Dtype>*>&bottom,
    				const vector<Blob<Dtype>*>&top);

    		Blob<unsigned int> map_;

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

    		int w_;
    		int h_;
    		int ch_size_;
    		int num_;
    };
} //namespace caffe

#endif // CAFFE_CUSTOM_LAYERS_HPP_
