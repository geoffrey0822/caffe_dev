/*
 * correlation_loss_layer.cpp
 *
 *  Created on: 2 Apr, 2018
 *      Author: gathetaroot
 */


#include "cfloat"
#include "caffe/customLayers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void caffe_pearson_correlation(const int N, const int k,
		const Dtype* x,const Dtype* y,Dtype* z){
	Dtype g1,g2,X,Y,meanX,meanY;
	for (int i = 0; i < N; i++){
			meanX = 0;
			meanY = 0;
			g1 = 0;
			g2 = 0;
			X = 0;
			Y = 0;
			for (int j = 0; j < k; j++){
				meanX += x[i*k + j] /k;
				meanY += y[i*k + j] / k;
			}
			for (int j = 0; j < k; j++){
				g1 += (x[i*k + j] - meanX)*(y[i*k + j] - meanY);
				X += (x[i*k + j] - meanX)*(x[i*k + j] - meanX);
				Y += (y[i*k + j] - meanY)*(y[i*k + j] - meanY);
			}
			printf("Mean:%f,%f\n", meanX, meanY);
			printf("G2X=%f,G2Y=%f\n", X, Y);
			g2 = sqrt(X*Y);
			z[i] = g1 / g2;
		}
}


template void caffe_pearson_correlation<double>(const int N, const int k,
		const double* x,const double* y,double* z);
template void caffe_pearson_correlation<float>(const int N, const int k,
		const float* x,const float* y,float* z);

template <typename Dtype>
void caffe_pearson_correlation_check(const int N, const int k,
		const Dtype* x,const Dtype* y,const Dtype* rx,const Dtype* ry,Dtype* z){
	Dtype g1,g2,X,Y,meanX,meanY;
	for (int i = 0; i < N; i++){
		if(rx[i]!=ry[i])
			z[i]=1;
		else{
			meanX = 0;
			meanY = 0;
			g1 = 0;
			g2 = 0;
			X = 0;
			Y = 0;
			for (int j = 0; j < k; j++){
				meanX += x[i*k + j] /k;
				meanY += y[i*k + j] / k;
			}
			for (int j = 0; j < k; j++){
				g1 += (x[i*k + j] - meanX)*(y[i*k + j] - meanY);
				X += (x[i*k + j] - meanX)*(x[i*k + j] - meanX);
				Y += (y[i*k + j] - meanY)*(y[i*k + j] - meanY);
			}
			printf("Mean:%f,%f\n", meanX, meanY);
			printf("G2X=%f,G2Y=%f\n", X, Y);
			g2 = sqrt(X*Y);
			z[i] = g1 / g2;
		}
	}
}

template void caffe_pearson_correlation_check<double>(const int N, const int k,
		const double* x,const double* y,const double* rx,const double* ry,double* z);
template void caffe_pearson_correlation_check<float>(const int N, const int k,
		const float* x,const float* y,const float* rx,const float* ry,float* z);

template <typename Dtype>
void caffe_diff_pearson_correlation(const int N, const int k,
		const Dtype* x, const Dtype* y, Dtype* dx,Dtype *dy){
	Dtype g1, g2, X, Y, meanX, meanY, xTerm, yTerm;
		for (int i = 0; i < N; i++){
			meanX = 0;
			meanY = 0;
			g1 = 0;
			g2 = 0;
			X = 0;
			Y = 0;
			xTerm = 0;
			yTerm = 0;
			for (int j = 0; j < k; j++){
				meanX += x[i*k + j] / k;
				meanY += y[i*k + j] / k;
			}
			for (int j = 0; j < k; j++){
				g1 += (x[i*k + j] - meanX)*(y[i*k + j] - meanY);
				Y += (y[i*k + j] - meanY)*(y[i*k + j] - meanY);
				X += (x[i*k + j] - meanX)*(x[i*k + j] - meanX);
				xTerm += (x[i*k + j] - meanX);
				yTerm += (y[i*k + j] - meanY);
			}
			g2 = sqrt(X*Y);
			for (int j = 0; j < k; j++){
				dx[i*k + j] = (k*(y[i*k + j] - meanY) / (g2)) - (( xTerm*Y*g1) / (g2*g2*g2));
				dy[i*k + j] = (k*(x[i*k + j] - meanX) / (g2)) - (( yTerm*X*g1) / (g2*g2*g2));
			}
		}
}


template void caffe_diff_pearson_correlation<double>(const int N, const int k,
		const double* x, const double* y, double* dx,double *dy);
template void caffe_diff_pearson_correlation<float>(const int N, const int k,
		const float* x, const float* y, float* dx,float *dy);


template<typename Dtype>
void CorrelationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>&bottom,
    	    				const vector<Blob<Dtype>*>&top){
	vector<int> in_shape=bottom[0]->shape();
	vector<int> in2_shape=bottom[1]->shape();

	_compareSame=false;
	CHECK(bottom.size()==2||bottom.size()==4)<<"The number of input must be either 2 or 4";
	if(bottom.size()>4){
		vector<int> in3_shape=bottom[2]->shape();
		vector<int> in4_shape=bottom[3]->shape();
		CHECK(in3_shape[1]==1&&in3_shape[1]==1)<<"3rd and 4th input should be the index of label dim=(Nx1)";

		_compareSame=true;
	}
	CHECK(in_shape.size()==in2_shape.size())<<"The shape of inputs must be the same";
	bool meetN1=false;
	bool meetN2=false;
	for(int i=1;i<in_shape.size();i++){
		if(in_shape[i]>1){
			CHECK(!meetN1)<<"Correlation Loss Function only support 1-dimensional data";
			meetN1=true;
		}
		if(in2_shape[i]>1){
			CHECK(!meetN2)<<"Correlation Loss Function only support 1-dimensional data";
			meetN2=true;
		}
	}

	vector<int> new_shape;
	new_shape.push_back(in_shape[0]);
	top[0]->Reshape(new_shape);

	vector<int> buff_bottom_shape;
	buff_bottom_shape.push_back(in_shape[0]);
	buff_bottom_shape.push_back(in_shape[1]);

	this->g1_.Reshape(new_shape);
	this->g2_.Reshape(new_shape);
	this->X_.Reshape(new_shape);
	this->Y_.Reshape(new_shape);
	this->x_mean_.Reshape(new_shape);
	this->y_mean_.Reshape(new_shape);
	this->xTerm_.Reshape(new_shape);
	this->yTerm_.Reshape(new_shape);
}

template<typename Dtype>
void CorrelationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>&bottom,
    	    				const vector<Blob<Dtype>*>&top){
	vector<int> in_shape=bottom[0]->shape();
	vector<int> out_shape=top[0]->shape();
	if(in_shape[0]!=out_shape[0]){
		vector<int> new_shape;
		new_shape.push_back(in_shape[0]);
		top[0]->Reshape(new_shape);
	}
}

template<typename Dtype>
void CorrelationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>&bottom,
        				const vector<Blob<Dtype>*>&top){
	vector<int>in_shape=bottom[0]->shape();
	int dim0=in_shape[0];
	int dim1=in_shape[1];
	const Dtype* x=bottom[0]->cpu_data();
	const Dtype* y=bottom[1]->cpu_data();

	Dtype* z=top[0]->mutable_cpu_data();

	if(_compareSame){
		const Dtype* rx=bottom[2]->cpu_data();
		const Dtype* ry=bottom[3]->cpu_data();
		caffe_pearson_correlation_check(dim0,dim1,x,y,rx,ry,z);
	}
	else
		caffe_pearson_correlation(dim0,dim1,x,y,z);
}

template<typename Dtype>
void CorrelationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>&top,
        				const vector<bool>& propagation_down,
    					const vector<Blob<Dtype>*>&bottom){
	vector<int>in_shape=bottom[0]->shape();
	int dim0=in_shape[0];
	int dim1=in_shape[1];
	const Dtype* x=bottom[0]->cpu_data();
	const Dtype* y=bottom[1]->cpu_data();
	Dtype* dgdx=bottom[0]->mutable_cpu_diff();
	Dtype* dgdy=bottom[1]->mutable_cpu_diff();
	const Dtype* delta=top[0]->cpu_diff();

	if(propagation_down[0]){
		caffe_diff_pearson_correlation(dim0,dim1,x,y,dgdx,dgdy);

		for(int i=0;i<dim0;i++){
			for(int j=0;j<dim1;j++){
				dgdx[i*dim1+j]*=delta[i];
			}
		}
	}else{
		int len=dim0*dim1;
		for(int i=0;i<len;i++){
			dgdx[i]=1;
		}
	}

}

#ifdef CPU_ONLY
STUB_GPU(CorrelationLayer);
#endif

INSTANTIATE_CLASS(CorrelationLayer);
REGISTER_LAYER_CLASS(Correlation);

}

