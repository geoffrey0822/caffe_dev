#ifndef CAFFE_CUSTOM_LAYERS_HPP_
#define CAFFE_CUSTOM_LAYERS_HPP_

#include <vector>
#include "caffe.hpp"
#include "blob.hpp"
#include "common.hpp"
#include "layer.hpp"
#include "loss_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

// Addition Mathematical Functions
template <typename Dtype>
void caffe_linear_proj(const int Nm,
		const Dtype* X,const unsigned int* Z,Dtype* Y);

template <typename Dtype>
void caffe_gpu_linear_proj(const int Nm,
		const Dtype* X,const unsigned int* Z,Dtype* Y);

/*--------Correlation Functions*/
template <typename Dtype>
void caffe_pearson_correlation(const int N, const int k,
		const Dtype* x,const Dtype* y,Dtype* z);

template <typename Dtype>
void caffe_pearson_correlation_check(const int N, const int k,
		const Dtype* x,const Dtype* y,const Dtype* rx,const Dtype* ry,Dtype* z);


template <typename Dtype>
void caffe_diff_pearson_correlation(const int N, const int k,
		const Dtype* x, const Dtype* y, Dtype* dx,Dtype *dy);


template <typename Dtype>
void caffe_gpu_pearson_correlation(const int N, const int k,
		const Dtype* x,const Dtype* y,Dtype* z,
		Dtype* x_mean,Dtype* y_mean,Dtype* g2_a,Dtype* g2_b,Dtype* g1);

template <typename Dtype>
void caffe_gpu_pearson_correlation_check(const int N, const int k,
		const Dtype* x,const Dtype* y,const Dtype* rx,const Dtype* ry,Dtype* z,
		Dtype* x_mean,Dtype* y_mean,Dtype* g2_a,Dtype* g2_b,Dtype* g1);

template <typename Dtype>
void caffe_gpu_diff_pearson_correlation(const int N, const int k,
		const Dtype* x, const Dtype* y, Dtype* dx,Dtype *dy,
		Dtype* X,Dtype* Y,
		Dtype* x_mean,Dtype* y_mean,Dtype* g2,Dtype* g1,
		Dtype* xTerm,Dtype *yTerm);

/*--------- Correlation Loss Functions---------------------*/
template <typename Dtype>
void gpu_exp_loss(const int N, const int ch, const Dtype* x, const Dtype scaler, Dtype* sumY);

template <typename Dtype>
void gpu_diff_exp_loss(const int N, const int ch, const Dtype* x, const Dtype scaler, Dtype* dx);

template <typename Dtype>
void caffe_exp_loss(const int N, const int ch, const Dtype* x, const Dtype scaler, Dtype* sumY);

template <typename Dtype>
void caffe_diff_exp_loss(const int N, const int ch, const Dtype*x, const Dtype scaler, Dtype* dx);

//

    template <typename Dtype>
    class MorphLayer: public Layer<Dtype>{
        public:
    		explicit MorphLayer(const LayerParameter& param):
			Layer<Dtype>(param){}

    		virtual inline const char* type() const {return "Morph";}

    		virtual inline int MinBottomBlobs()const{return 2;}
    		virtual inline int MaxBottomBlobs()const{return 2;}
    		virtual inline int MinTopBlobs()const{return 1;}
    		virtual inline int MaxTopBlobs()const{return 1;}
    		virtual inline int ExactNumBottomBlobs() const { return 2; }
    		virtual inline int ExactNumTopBlobs() const { return 1; }

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
    template <typename Dtype>
    class CorrelationLayer:public Layer<Dtype>{
    public:
    	explicit CorrelationLayer(const LayerParameter& param):
    	Layer<Dtype>(param){}
    	virtual inline const char* type() const {return "Correlation";}
    	virtual inline int MinBottomBlobs()const{return 2;}
    	virtual inline int MaxBottomBlobs()const{return 4;}
    	virtual inline int ExactNumTopBlobs() const { return 1; }
    	virtual void LayerSetUp(const vector<Blob<Dtype>*>&bottom,
    	    				const vector<Blob<Dtype>*>&top);

    	virtual void Reshape(const vector<Blob<Dtype>*>&bottom,
    	    				const vector<Blob<Dtype>*>&top);

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
        Dtype _threshold;
        bool _norm;
        bool _compareSame;
        Blob<Dtype> x_mean_;
        Blob<Dtype> y_mean_;
        Blob<Dtype> X_;
        Blob<Dtype> Y_;
        Blob<Dtype> g1_;
        Blob<Dtype> g2_;
        Blob<Dtype> xTerm_;
        Blob<Dtype> yTerm_;

    };

    template <typename Dtype>
        class ExpDecayLossLayer:public Layer<Dtype>{
        public:
        	explicit ExpDecayLossLayer(const LayerParameter& param):
        	Layer<Dtype>(param){}
        	virtual inline const char* type() const {return "Exponential Decay Loss";}
        	virtual inline int ExactNumBottomBlobs() const { return 1; }
        	virtual inline int ExactNumTopBlobs() const { return 1; }
        	virtual void LayerSetUp(const vector<Blob<Dtype>*>&bottom,
        	    				const vector<Blob<Dtype>*>&top);

        	virtual void Reshape(const vector<Blob<Dtype>*>&bottom,
        	    				const vector<Blob<Dtype>*>&top);

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
            Dtype _scaler;

        };
} //namespace caffe

#endif // CAFFE_CUSTOM_LAYERS_HPP_
