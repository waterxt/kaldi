//nnet/nnet-nnet-multi-gpu.cc
// CopyRight: tao xu
// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-affine-transform.h"
#include "nnet/nnet-linear-transform.h"
#include "nnet/nnet-convolutional-component.h"
#include "nnet/nnet-convolutional-2d-component.h"
#include "nnet/nnet-lstm-projected.h"
#include "nnet/nnet-blstm-projected.h"
#include "nnet/nnet-recurrent.h"
#include "nnet/nnet-parametric-relu.h"
#include "cudamatrix/cu-common.h"


namespace kaldi{
namespace nnet1{

//multi-gpu training
int32 Nnet::GetDim()const{

  int32 dim = 0;
  AffineTransform* affine_p ;
  LinearTransform* line_p ;
  BlstmProjected* pblstm_p ; 
  Convolutional2DComponent* con2d_p ;
  ConvolutionalComponent* con1d_p ;
  LstmProjected* plstm_p ;
  ParametricRelu* pRelu_p ;
  RecurrentComponent* rnn_p ;
  for(int32 i = 0; i < components_.size(); i++){

  	if(components_[i]->IsUpdatable()){
  		switch(components_[i]->GetType()){
  			case Component::kAffineTransform:
  				affine_p = (AffineTransform*)(components_[i]);
  				dim += affine_p->linearity_.SizeInBytes()/sizeof(BaseFloat);
  				dim += affine_p->bias_.Dim();
  				break;
  			case Component::kLinearTransform:
  				line_p = (LinearTransform*)(components_[i]);
  				dim += line_p->linearity_.SizeInBytes()/sizeof(BaseFloat);
  				break;
  			case Component::kBlstmProjected:
  				pblstm_p = (BlstmProjected*)(components_[i]);
  				dim += pblstm_p->f_w_gifo_x_.SizeInBytes()/sizeof(BaseFloat);
  				dim += pblstm_p->b_w_gifo_x_.SizeInBytes()/sizeof(BaseFloat);
  				dim += pblstm_p->f_w_gifo_r_.SizeInBytes()/sizeof(BaseFloat);
  				dim += pblstm_p->b_w_gifo_r_.SizeInBytes()/sizeof(BaseFloat);
  				dim += pblstm_p->f_bias_.Dim();
  				dim += pblstm_p->b_bias_.Dim();
  				dim += pblstm_p->f_peephole_i_c_.Dim();
  				dim += pblstm_p->f_peephole_f_c_.Dim();
  				dim += pblstm_p->f_peephole_o_c_.Dim();
  				dim += pblstm_p->b_peephole_i_c_.Dim();
  				dim += pblstm_p->b_peephole_f_c_.Dim();
  				dim += pblstm_p->b_peephole_o_c_.Dim();
  				dim += pblstm_p->f_w_r_m_.SizeInBytes()/sizeof(BaseFloat);
  				dim += pblstm_p->b_w_r_m_.SizeInBytes()/sizeof(BaseFloat);
  				break;
  			case Component::kConvolutional2DComponent:
  				con2d_p = (Convolutional2DComponent*)(components_[i]);
  				dim += con2d_p->filters_.SizeInBytes()/sizeof(BaseFloat);
  				dim += con2d_p->bias_.Dim();
  				break;
  			case Component::kConvolutionalComponent:
  				con1d_p = (ConvolutionalComponent*)(components_[i]);
  				dim += con1d_p->filters_.SizeInBytes()/sizeof(BaseFloat);
  				dim += con1d_p->bias_.Dim();
  				break;
  			case Component::kLstmProjected:
  				plstm_p = (LstmProjected*)(components_[i]);
  				dim += plstm_p->w_gifo_x_.SizeInBytes()/sizeof(BaseFloat);
  				dim += plstm_p->w_gifo_r_.SizeInBytes()/sizeof(BaseFloat);
  				dim += plstm_p->bias_.Dim();
  				dim += plstm_p->peephole_i_c_.Dim();
  				dim += plstm_p->peephole_f_c_.Dim();
  				dim += plstm_p->peephole_o_c_.Dim();
  				dim += plstm_p->w_r_m_.SizeInBytes()/sizeof(BaseFloat);
  				break;
  			case Component::kParametricRelu:
  				pRelu_p = (ParametricRelu*)(components_[i]);
  				dim += pRelu_p->alpha_.Dim();
  				dim += pRelu_p->beta_.Dim();
  				break;
  			case Component::kRecurrentComponent:
  				rnn_p = (RecurrentComponent*)(components_[i]);
  				dim += rnn_p->w_forward_.SizeInBytes()/sizeof(BaseFloat);
  				dim += rnn_p->w_recurrent_.SizeInBytes()/sizeof(BaseFloat);
  				dim += rnn_p->bias_.Dim();
  				break;
  			default:
  				KALDI_ERR<<" unimplement component : "<<Component::TypeToMarker(components_[i]->GetType());
  		}
  	}


  }
  return dim;

}

void Nnet::InitData(){

	if(NULL != this->free_data_)
		return;
	size_t size = 0 ;
	void* data = NULL;
	void* free_data = NULL;
	int32 dim = 0;
	dim = this->GetDim();
	size = dim * sizeof(BaseFloat)+16;
	CU_SAFE_CALL(cudaMalloc((void**)&free_data_, size));
	data = (free_data_ ? (void *)( (((unsigned long)*(&free_data)) + 15) & ~0xFUL ) : NULL) ;

	if(NULL != data){
		this->data_ = static_cast<BaseFloat*>(data);
		this->free_data_ = static_cast<BaseFloat*>(free_data);
	}else{
		throw std::bad_alloc();
	}
}

void Nnet::GetWeights(){

  int32 pos = 0 ;
  int32 size = 0 ;
  MatrixDim dim;
  int32 src_pitch, dst_pitch, width ;
  AffineTransform* affine_p ;
  LinearTransform* line_p ;
  BlstmProjected* pblstm_p ; 
  Convolutional2DComponent* con2d_p ;
  ConvolutionalComponent* con1d_p ;
  LstmProjected* plstm_p ;
  ParametricRelu* pRelu_p ;
  RecurrentComponent* rnn_p ;
  for(int32 i = 0; i < components_.size(); i++){
  	if(components_[i]->IsUpdatable()){
  		switch(components_[i]->GetType()){
  			case Component::kAffineTransform:
  			affine_p = (AffineTransform*)(components_[i]);
  			dim = affine_p->linearity_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D((uint8_t*)data_+pos, dst_pitch, affine_p->linearity_.Data(), 
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += affine_p->linearity_.SizeInBytes();
  			size = affine_p->bias_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy((uint8_t*)data_+pos, affine_p->bias_.Data(), size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			break;
  			case Component::kLinearTransform:
  			line_p = (LinearTransform*)(components_[i]);
  			dim = line_p->linearity_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D((uint8_t*)data_+pos, dst_pitch, line_p->linearity_.Data(),
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += line_p->linearity_.SizeInBytes() ;
  			break;
  			case Component::kBlstmProjected:
  			pblstm_p = (BlstmProjected*)(components_[i]);
  			dim = pblstm_p->f_w_gifo_x_.Dim() ;
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D((uint8_t*)data_+pos, dst_pitch, pblstm_p->f_w_gifo_x_.Data(),
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += pblstm_p->f_w_gifo_x_.SizeInBytes() ;
  			dim = pblstm_p->b_w_gifo_x_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D((uint8_t*)data_+pos, dst_pitch, pblstm_p->b_w_gifo_x_.Data(),
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += pblstm_p->b_w_gifo_x_.SizeInBytes();
  			dim = pblstm_p->f_w_gifo_r_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D((uint8_t*)data_+pos, dst_pitch, pblstm_p->f_w_gifo_r_.Data(),
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += pblstm_p->f_w_gifo_r_.SizeInBytes();
  			dim = pblstm_p->b_w_gifo_r_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D((uint8_t*)data_+pos, dst_pitch, pblstm_p->b_w_gifo_r_.Data(),
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += pblstm_p->b_w_gifo_r_.SizeInBytes();
  			size = pblstm_p->f_bias_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy((uint8_t*)data_+pos, pblstm_p->f_bias_.Data(), size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			size = pblstm_p->b_bias_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy((uint8_t*)data_+pos, pblstm_p->b_bias_.Data(), size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			size = pblstm_p->f_peephole_i_c_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy((uint8_t*)data_+pos, pblstm_p->f_peephole_i_c_.Data(), size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			size = pblstm_p->f_peephole_o_c_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy((uint8_t*)data_+pos, pblstm_p->f_peephole_o_c_.Data(), size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			size = pblstm_p->f_peephole_f_c_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy((uint8_t*)data_+pos, pblstm_p->f_peephole_f_c_.Data(), size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			dim = pblstm_p->f_w_r_m_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D((uint8_t*)data_+pos, dst_pitch, pblstm_p->f_w_r_m_.Data(),
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += pblstm_p->f_w_r_m_.SizeInBytes();
  			dim = pblstm_p->b_w_r_m_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D((uint8_t*)data_+pos, dst_pitch, pblstm_p->b_w_r_m_.Data(),
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += pblstm_p->b_w_r_m_.SizeInBytes();
  			break;  	
  			case Component::kConvolutional2DComponent:
  			con2d_p = (Convolutional2DComponent*)(components_[i]);
  			dim = con2d_p->filters_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);	
  			CU_SAFE_CALL(cudaMemcpy2D((uint8_t*)data_+pos, dst_pitch, con2d_p->filters_.Data(),
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += con2d_p->filters_.SizeInBytes();
  			size = con2d_p->bias_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy((uint8_t*)data_+pos, con2d_p->bias_.Data(), size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			break;
  			case Component::kConvolutionalComponent:
  			con1d_p = (ConvolutionalComponent*)(components_[i]);
  			dim = con1d_p->filters_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D((uint8_t*)data_+pos, dst_pitch, con1d_p->filters_.Data(),
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += con1d_p->filters_.SizeInBytes();
  			size = con1d_p->bias_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy((uint8_t*)data_+pos, con1d_p->bias_.Data(), size, cudaMemcpyDeviceToDevice));
  			pos += size; 
  			break;
  			case Component::kLstmProjected:
  			plstm_p = (LstmProjected*)(components_[i]);
  			dim = plstm_p->w_gifo_x_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D((uint8_t*)data_ + pos, dst_pitch, plstm_p->w_gifo_x_.Data(),
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += plstm_p->w_gifo_x_.SizeInBytes();
  			dim = plstm_p->w_gifo_r_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D((uint8_t*)data_+pos, dst_pitch, plstm_p->w_gifo_r_.Data(),
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += plstm_p->w_gifo_r_.SizeInBytes();
  			size = plstm_p->bias_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy((uint8_t*)data_+pos, plstm_p->bias_.Data(), size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			size = plstm_p->peephole_i_c_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy((uint8_t*)data_+pos, plstm_p->peephole_i_c_.Data(), size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			size = plstm_p->peephole_o_c_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy((uint8_t*)data_+pos, plstm_p->peephole_o_c_.Data(), size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			size = plstm_p->peephole_f_c_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy((uint8_t*)data_+pos, plstm_p->peephole_f_c_.Data(), size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			dim = plstm_p->w_r_m_.Dim();
  			src_pitch = dim.stride*sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D((uint8_t*)data_+pos, dst_pitch, plstm_p->w_r_m_.Data(),
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += plstm_p->w_r_m_.SizeInBytes();
  			break;
  			case Component::kParametricRelu:
  			pRelu_p = (ParametricRelu*)(components_[i]);
  			size = pRelu_p->alpha_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy((uint8_t*)data_+pos, pRelu_p->alpha_.Data(), size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			size = pRelu_p->beta_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy((uint8_t*)data_+pos, pRelu_p->beta_.Data(), size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			break;
  			case Component::kRecurrentComponent:
  			rnn_p = (RecurrentComponent*)(components_[i]);
  			dim = rnn_p->w_forward_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D((uint8_t*)data_+pos, dst_pitch, rnn_p->w_forward_.Data(),
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += rnn_p->w_forward_.SizeInBytes();
  			dim = rnn_p->w_recurrent_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D((uint8_t*)data_+pos, dst_pitch, rnn_p->w_recurrent_.Data(),
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += rnn_p->w_recurrent_.SizeInBytes();
  			size = rnn_p->bias_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy((uint8_t*)data_+pos, rnn_p->bias_.Data(), size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			break;
  			default:

  			KALDI_ERR<<" unimplement component : "<<Component::TypeToMarker(components_[i]->GetType());
  		}

  }
}

}

void Nnet::SetWeights(){
  int32 pos = 0 ;
  int32 size = 0 ;
  MatrixDim dim;
  int32 src_pitch, dst_pitch, width ;
  AffineTransform* affine_p ;
  LinearTransform* line_p ;
  BlstmProjected* pblstm_p ; 
  Convolutional2DComponent* con2d_p ;
  ConvolutionalComponent* con1d_p ;
  LstmProjected* plstm_p ;
  ParametricRelu* pRelu_p ;
  RecurrentComponent* rnn_p ;
  for(int32 i = 0; i < components_.size(); i++){
  	switch(components_[i]->GetType()){
  		case Component::kAffineTransform:
  			affine_p = (AffineTransform*)(components_[i]);
  			dim = affine_p->linearity_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D(affine_p->linearity_.Data(), dst_pitch, (uint8_t*)data_+pos, 
  									src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += affine_p->linearity_.SizeInBytes();
  			size = affine_p->bias_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy(affine_p->bias_.Data(), (uint8_t*)data_+pos, size, cudaMemcpyDeviceToDevice));
  			break;
  		case Component::kLinearTransform:
  			line_p = (LinearTransform*)(components_[i]);
  			dim = affine_p->linearity_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D(line_p->linearity_.Data(), dst_pitch, (uint8_t*)data_+pos,
  									src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += line_p->linearity_.SizeInBytes();
  			break;
  		case Component::kBlstmProjected:
  			pblstm_p = (BlstmProjected*)(components_[i]);
  			dim = pblstm_p->f_w_gifo_x_.Dim() ;
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D(pblstm_p->f_w_gifo_x_.Data(), dst_pitch, (uint8_t*)data_+pos,
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += pblstm_p->f_w_gifo_x_.SizeInBytes() ;
  			dim = pblstm_p->b_w_gifo_x_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D(pblstm_p->b_w_gifo_x_.Data(), dst_pitch, (uint8_t*)data_+pos,
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += pblstm_p->b_w_gifo_x_.SizeInBytes();
  			dim = pblstm_p->f_w_gifo_r_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D(pblstm_p->f_w_gifo_r_.Data(), dst_pitch, (uint8_t*)data_+pos,
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += pblstm_p->f_w_gifo_r_.SizeInBytes();
  			dim = pblstm_p->b_w_gifo_r_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D(pblstm_p->b_w_gifo_r_.Data(),dst_pitch, (uint8_t*)data_+pos,
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += pblstm_p->b_w_gifo_r_.SizeInBytes();
  			size = pblstm_p->f_bias_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy( pblstm_p->f_bias_.Data(), (uint8_t*)data_+pos, size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			size = pblstm_p->b_bias_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy( pblstm_p->f_bias_.Data(), (uint8_t*)data_+pos,  size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			size = pblstm_p->f_peephole_i_c_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy(pblstm_p->f_peephole_i_c_.Data(), (uint8_t*)data_+pos, size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			size = pblstm_p->f_peephole_o_c_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy(pblstm_p->f_peephole_o_c_.Data(), (uint8_t*)data_+pos, size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			size = pblstm_p->f_peephole_f_c_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy(pblstm_p->f_peephole_f_c_.Data(), (uint8_t*)data_+pos,  size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			dim = pblstm_p->f_w_r_m_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D(pblstm_p->f_w_r_m_.Data(), dst_pitch, (uint8_t*)data_+pos,
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += pblstm_p->f_w_r_m_.SizeInBytes();
  			dim = pblstm_p->b_w_r_m_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D(pblstm_p->b_w_r_m_.Data(), dst_pitch, (uint8_t*)data_+pos,
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += pblstm_p->b_w_r_m_.SizeInBytes();
  			break;
  		case Component::kConvolutional2DComponent:
  			con2d_p = (Convolutional2DComponent*)(components_[i]);
  			dim = con2d_p->filters_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);	
  			CU_SAFE_CALL(cudaMemcpy2D(con2d_p->filters_.Data(), dst_pitch, (uint8_t*)data_+pos,
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += con2d_p->filters_.SizeInBytes();
  			size = con2d_p->bias_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy(con2d_p->bias_.Data(), (uint8_t*)data_+pos,  size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			break;
  		case Component::kConvolutionalComponent:
  			con1d_p = (ConvolutionalComponent*)(components_[i]);
  			dim = con1d_p->filters_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D( con1d_p->filters_.Data(), dst_pitch,(uint8_t*)data_+pos,
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += con1d_p->filters_.SizeInBytes();
  			size = con1d_p->bias_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy( con1d_p->bias_.Data(), (uint8_t*)data_+pos, size, cudaMemcpyDeviceToDevice));
  			pos += size; 
  			break;
  		case Component::kLstmProjected:
  			plstm_p = (LstmProjected*)(components_[i]);
  			dim = plstm_p->w_gifo_x_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D(plstm_p->w_gifo_x_.Data(), dst_pitch, (uint8_t*)data_ + pos,
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += plstm_p->w_gifo_x_.SizeInBytes();
  			dim = plstm_p->w_gifo_r_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D(plstm_p->w_gifo_r_.Data(), dst_pitch, (uint8_t*)data_+pos, 
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += plstm_p->w_gifo_r_.SizeInBytes();
  			size = plstm_p->bias_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy(plstm_p->bias_.Data(), (uint8_t*)data_+pos,  size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			size = plstm_p->peephole_i_c_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy(plstm_p->peephole_i_c_.Data(), (uint8_t*)data_+pos,  size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			size = plstm_p->peephole_o_c_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy(plstm_p->peephole_o_c_.Data(), (uint8_t*)data_+pos,  size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			size = plstm_p->peephole_f_c_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy(plstm_p->peephole_f_c_.Data(), (uint8_t*)data_+pos,  size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			dim = plstm_p->w_r_m_.Dim();
  			src_pitch = dim.stride*sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D( plstm_p->w_r_m_.Data(), dst_pitch, (uint8_t*)data_+pos,
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += plstm_p->w_r_m_.SizeInBytes();
  			break;
  		case Component::kParametricRelu:
  			pRelu_p = (ParametricRelu*)(components_[i]);
  			size = pRelu_p->alpha_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy(pRelu_p->alpha_.Data(), (uint8_t*)data_+pos,  size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			size = pRelu_p->beta_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy(pRelu_p->beta_.Data(), (uint8_t*)data_+pos,  size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			break;
  		case Component::kRecurrentComponent:
  			rnn_p = (RecurrentComponent*)(components_[i]);
  			dim = rnn_p->w_forward_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch ;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D(rnn_p->w_forward_.Data(), dst_pitch, (uint8_t*)data_+pos,
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += rnn_p->w_forward_.SizeInBytes();
  			dim = rnn_p->w_recurrent_.Dim();
  			src_pitch = dim.stride * sizeof(BaseFloat);
  			dst_pitch = src_pitch;
  			width = dim.cols * sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy2D(rnn_p->w_recurrent_.Data(), dst_pitch, (uint8_t*)data_+pos, 
  				src_pitch, width, dim.rows, cudaMemcpyDeviceToDevice));
  			pos += rnn_p->w_recurrent_.SizeInBytes();
  			size = rnn_p->bias_.Dim()*sizeof(BaseFloat);
  			CU_SAFE_CALL(cudaMemcpy(rnn_p->bias_.Data(), (uint8_t*)data_+pos,  size, cudaMemcpyDeviceToDevice));
  			pos += size ;
  			break;
  		default:

  			KALDI_ERR<<" unimplement component : "<<Component::TypeToMarker(components_[i]->GetType());



  	}

  }


}

}

}