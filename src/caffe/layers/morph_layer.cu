#include <vector>

#include "caffe/customLayers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void proj_kernel(const int n,const Dtype* x,const unsigned int* z,
		Dtype* y){
	CUDA_KERNEL_LOOP(i,n){
		y[i]=x[z[i]];
	}
}

template <typename Dtype>
void caffe_gpu_linear_proj(const int Nm,
		const Dtype* X,const unsigned int* Z,Dtype* Y){
	proj_kernel<Dtype><<<CAFFE_GET_BLOCKS(Nm),CAFFE_CUDA_NUM_THREADS>>>(Nm,
			X,Z,Y);
}

template void caffe_gpu_linear_proj<int>(const int Nm,
		const int* X,const unsigned int* Z,int* Y);
template void caffe_gpu_linear_proj<double>(const int Nm,
		const double* X,const unsigned int* Z,double* Y);
template void caffe_gpu_linear_proj<float>(const int Nm,
		const float* X,const unsigned int* Z,float* Y);

template <typename Dtype>
void MorphLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>&bottom,
                             const vector<Blob<Dtype>*>&top){
	const Dtype* bottom1_data=bottom[0]->gpu_data();
	const Dtype* bottom2_data=bottom[1]->gpu_data();
	Dtype* top1_data=top[0]->mutable_gpu_data();
	const unsigned int* map_data=map_.gpu_data();
	Dtype* top2_data=top[1]->mutable_gpu_data();

	size_t count=bottom[0]->count();

	caffe_copy(count,bottom1_data,top1_data);
	caffe_gpu_linear_proj(count,bottom2_data,map_data,top2_data);
	CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
void MorphLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>&top,
                               const vector<bool>& propagation_down,
                              const vector<Blob<Dtype>*>&bottom){
	Dtype* bottom1_diff=bottom[0]->mutable_gpu_diff();
	Dtype* bottom2_diff=bottom[1]->mutable_gpu_diff();
	const Dtype* top1_diff=top[0]->gpu_diff();

	caffe_copy(bottom[0]->count(),top1_diff,bottom1_diff);
	caffe_gpu_set(bottom[1]->count(),Dtype(0),bottom2_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(MorphLayer);

}


