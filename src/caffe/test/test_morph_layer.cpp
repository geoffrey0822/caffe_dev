/*
 * test_morph_layer.cpp
 *
 *  Created on: 11 Mar, 2018
 *      Author: gathetaroot
 */
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/customLayers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe{

template <typename TypeParam>
class MorphLayer_Test: public MultiDeviceTest<TypeParam>{
	typedef typename TypeParam::Dtype Dtype;

protected:
	MorphLayer_Test()
		:bblob_0_(new Blob<Dtype>(32,14,1,1)),
		 bblob_1_(new Blob<Dtype>(32,2,1,1)),
		 bblob_2_(new Blob<Dtype>(32,14,10,10)),
		 bblob_3_(new Blob<Dtype>(32,2,10,10)),
		 bblob_4_(new Blob<Dtype>(32,14,1,1)),
		 bblob_5_(new Blob<Dtype>(32,2,1,1)),
		 blob_top_0_(new Blob<Dtype>),
		 blob_top_1_(new Blob<Dtype>){}

	virtual void SetUp(){

		vector<int> shape1(2);
		vector<int> shape2(2);
		shape1.push_back(32);
		shape1.push_back(14);
		shape2.push_back(32);
		shape2.push_back(2);
		bblob_4_->Reshape(shape1);
		bblob_5_->Reshape(shape2);

		caffe_rng_uniform<Dtype>(bblob_0_->count(),Dtype(0),Dtype(1),bblob_0_->mutable_cpu_data());
		caffe_rng_uniform<Dtype>(bblob_1_->count(),Dtype(0),Dtype(1),bblob_1_->mutable_cpu_data());
		caffe_rng_uniform<Dtype>(bblob_2_->count(),Dtype(0),Dtype(1),bblob_2_->mutable_cpu_data());
		caffe_rng_uniform<Dtype>(bblob_3_->count(),Dtype(0),Dtype(1),bblob_3_->mutable_cpu_data());
		caffe_rng_uniform<Dtype>(bblob_4_->count(),Dtype(0),Dtype(1),bblob_4_->mutable_cpu_data());
		caffe_rng_uniform<Dtype>(bblob_5_->count(),Dtype(0),Dtype(1),bblob_5_->mutable_cpu_data());

		bblobs_.push_back(bblob_0_);
		bblobs_.push_back(bblob_1_);
		bblobs_md_.push_back(bblob_2_);
		bblobs_md_.push_back(bblob_3_);
		bblobs_0d_.push_back(bblob_4_);
		bblobs_0d_.push_back(bblob_5_);
		tblobs_.push_back(blob_top_0_);
		tblobs_.push_back(blob_top_1_);
	}

	virtual ~MorphLayer_Test(){
		delete bblob_0_;
		delete bblob_1_;
		delete bblob_2_;
		delete bblob_3_;
		delete bblob_4_;
		delete bblob_5_;
		delete blob_top_0_;
		delete blob_top_1_;
	}

	Blob<Dtype>* const bblob_0_;
	Blob<Dtype>* const bblob_1_;
	Blob<Dtype>* const bblob_2_;
	Blob<Dtype>* const bblob_3_;
	Blob<Dtype>* bblob_4_;
	Blob<Dtype>* bblob_5_;
	Blob<Dtype>* const blob_top_0_;
	Blob<Dtype>* const blob_top_1_;
	vector<Blob<Dtype>*> bblobs_;
	vector<Blob<Dtype>*> bblobs_md_;
	vector<Blob<Dtype>*> bblobs_0d_;
	vector<Blob<Dtype>*> tblobs_;

};

TYPED_TEST_CASE(MorphLayer_Test,TestDtypesAndDevices);
TYPED_TEST(MorphLayer_Test, TestSetupNum){
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	MorphLayer<Dtype> layer(layer_param);
	layer.SetUp(this->bblobs_,this->tblobs_);

	EXPECT_EQ(this->bblobs_[0]->shape()[0],this->tblobs_[0]->shape()[0]);
	EXPECT_EQ(this->bblobs_[0]->shape()[1],this->tblobs_[0]->shape()[1]);
	EXPECT_EQ(this->bblobs_[0]->shape()[2],this->tblobs_[0]->shape()[2]);
	EXPECT_EQ(this->bblobs_[0]->shape()[3],this->tblobs_[0]->shape()[3]);

	EXPECT_EQ(this->bblobs_[0]->shape()[0],this->tblobs_[1]->shape()[0]);
	EXPECT_EQ(this->bblobs_[0]->shape()[1],this->tblobs_[1]->shape()[1]);
	EXPECT_EQ(this->bblobs_[0]->shape()[2],this->tblobs_[1]->shape()[2]);
	EXPECT_EQ(this->bblobs_[0]->shape()[3],this->tblobs_[1]->shape()[3]);

}
TYPED_TEST(MorphLayer_Test, TestForward){
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	MorphLayer<Dtype> layer(layer_param);
	layer.SetUp(this->bblobs_,this->tblobs_);
	layer.Forward(this->bblobs_,this->tblobs_);

	for(int i=0;i<this->tblobs_[0]->count();i++){
		EXPECT_EQ(this->bblobs_[0]->cpu_data()[i],this->tblobs_[0]->cpu_data()[i]);
	}
	for(int i=0;i<this->tblobs_[1]->count();i++){
		EXPECT_EQ(this->bblobs_[1]->cpu_data()[layer.map_.cpu_data()[i]],this->tblobs_[1]->cpu_data()[i]);
	}
}

TYPED_TEST(MorphLayer_Test, TestForward_MD){
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	MorphLayer<Dtype> layer(layer_param);
	layer.SetUp(this->bblobs_md_,this->tblobs_);
	layer.Forward(this->bblobs_md_,this->tblobs_);

	for(int i=0;i<this->tblobs_[0]->count();i++){
		EXPECT_EQ(this->bblobs_md_[0]->cpu_data()[i],this->tblobs_[0]->cpu_data()[i]);
	}
	for(int i=0;i<this->tblobs_[1]->count();i++){
		EXPECT_EQ(this->bblobs_md_[1]->cpu_data()[layer.map_.cpu_data()[i]],this->tblobs_[1]->cpu_data()[i]);
	}
}


TYPED_TEST(MorphLayer_Test, TestBackward){
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	MorphLayer<Dtype> layer(layer_param);
	GradientChecker<Dtype> checker(1e-2,1e-2);
	checker.CheckGradient(&layer,this->bblobs_,this->tblobs_,0);
}

TYPED_TEST(MorphLayer_Test, TestBackward_MD){
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	MorphLayer<Dtype> layer(layer_param);
	//GradientChecker<Dtype> checker(1e-3,1e-6);
	//checker.CheckGradient(&layer,this->bblobs_md_,this->tblobs_,0);
}


} // namespace caffe



