//nnet/nnet-update-parallel.cc
//CopyRight 2017 taoxu


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
#include <numeric>
#include <nccl.h>
#include "nnet/nnet-update-parallel.h"
#include "util/kaldi-thread.h"
#include "nnet/nnet-example.h"
#include "base/timer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"

namespace kaldi{
namespace nnet1{

class DoBackpropParallelClass: public MultiThreadable {
 public:
  // This constructor is only called for a temporary object
  // that we pass to the RunMultiThreaded function.
  DoBackpropParallelClass(const Nnet &nnet,
                          ExamplesRepository *repository,
                          std::string objective_function,
                          double *tot_weight_ptr,
                          double *log_prob_ptr,
                          bool crossvalidate):
      nnet_(nnet), repository_(repository),
      objective_function_(objective_function),
      tot_weight_ptr_(tot_weight_ptr),
      log_prob_ptr_(log_prob_ptr),
      tot_weight_(0.0),
      log_prob_(0.0),
      crossvalidate_(crossvalidate) { }

  // This does the main function of the class.
  void operator () () {

  	int nDev = 0;
  	kaldi::int64 total_frames = 0;
  	cudaGetDeviceCount(&nDev);
  	cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
  	ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*nDev);
  	int devs[4] = { 0, 1, 2, 3 };
  	NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
  	NCCLCHECK(ncclCommCount(comms[0], &nDev));
  	int32 nnet_bytes = nnet_.GetDim()*sizeof(BaseFloat);
  	Nnet* nnet = (Nnet*)malloc(sizeof(Nnet)*nDev);
  	Nnet nnet_reduce ;
  	CuMatrix<BaseFloat>* nnet_in = (CuMatrix<BaseFloat>*)malloc(sizeof(CuMatrix<BaseFloat>)*nDev);
  	CuMatrix<BaseFloat>* nnet_out = (CuMatrix<BaseFloat>*)malloc(sizeof(CuMatrix<BaseFloat>)*nDev);
  	CuMatrix<BaseFloat>* obj_diff = (CuMatrix<BaseFloat>*)malloc(sizeof(CuMatrix<BaseFloat>)*nDev);
  	Vector<BaseFloat>* frame_weights = (Vector<BaseFloat>*)malloc(sizeof(CuMatrix<BaseFloat>)*nDev);
  	Posterior* nnet_tgt = (Posterior*)malloc(sizeof(Posterior)*nDev);
  	NnetExample* example = (NnetExample*)malloc(sizeof(NnetExample)*nDev);
    Xent xent;
    Mse mse;
    MultiTaskLoss multitask;
  	for (int i = 0; i < nDev; ++i) {

    	cudaSetDevice(i);
    	cudaStreamCreate(s+i);
    	nnet[i] = nnet_;
    }

    nnet_reduce = nnet_ ;
    bool done = false;

    while (!done) {
    	for (int i = 0; i < nDev; ++i) {

    		cudaSetDevice(i);
    		if(repository_->ProvideExamples(example[i])){
    			done = true ;
    			continue;
    		}
    		nnet_in[i] = *(example[i].mat);
        	nnet_tgt[i] = *(example[i].targets);
        	frame_weights[i] = *(example[i].frame_weights);

    		nnet[i].Propagate(nnet_in[i], nnet_out+i);
        	// evaluate objective function we've chosen,
    		if (objective_function_ == "xent") {
          	// gradients re-scaled by weights in Eval,
    			xent.Eval(frame_weights[i], nnet_out[i], nnet_tgt[i], obj_diff+i);
    		} else if (objective_function_ == "mse") {
          	// gradients re-scaled by weights in Eval,
    			mse.Eval(frame_weights[i], nnet_out[i], nnet_tgt[i], obj_diff+i);
    		} else if (0 == objective_function_.compare(0, 9, "multitask")) {
          	// gradients re-scaled by weights in Eval,
    			multitask.Eval(frame_weights[i], nnet_out[i], nnet_tgt[i], obj_diff+i);
    		} else {
    			KALDI_ERR << "Unknown objective function code : " << objective_function_;
    		}

    		if (!crossvalidate_) {
          	// back-propagate, and do the update,
    			nnet[i].Backpropagate(obj_diff[i], NULL);
    		}

    		if( i == 0){
    			nnet[i].GetWeights();
    			NCCLCHECK(ncclReduce((const void*)nnet[i].Data(), (void*)nnet_reduce.Data(), nnet_bytes,
    				ncclChar, ncclSum, 0, comms[i], s[i]));

    		}else{
    			nnet[i].GetWeights();
    			NCCLCHECK(ncclReduce((const void*)nnet[i].Data(), NULL, nnet_bytes,
    				ncclChar, ncclSum, 0, comms[i], s[i]));
    		}

    		cudaStreamSynchronize(s[i]);
  			
    		if(i == 0)
    			nnet[0] = nnet_reduce ;

    		NCCLCHECK(ncclBcast((void*)nnet[i].Data(), nnet_bytes, ncclChar, 0, comms[i], s[i]));
    		nnet[i].SetWeights();

    		// 1st mini-batch : show what happens in network,
        	if (total_frames == 0) {
        		KALDI_VLOG(1) << "### After " << total_frames << " frames,";
        		KALDI_VLOG(1) << nnet[i].InfoPropagate();

        		if (!crossvalidate_) {
          			KALDI_VLOG(1) << nnet[i].InfoBackPropagate();
          			KALDI_VLOG(1) << nnet[i].InfoGradient();
        		}
        	}
    	}

        if (GetVerboseLevel() >= 2) {
          static int32 counter = 0;
          cudaSetDevice(0);
          counter += nDev * nnet_in[0].NumRows();
          // print every 25k frames,
          if (counter >= 25000) {
            KALDI_VLOG(2) << "### After " << total_frames << " frames,";
            for(int i = 0; i < nDev; i++){

    			cudaSetDevice(i);
            	KALDI_VLOG(2) << nnet[i].InfoPropagate();
            	if (!crossvalidate_) {
              		KALDI_VLOG(2) << nnet[i].InfoBackPropagate();
              		KALDI_VLOG(2) << nnet[i].InfoGradient();
            	}
            }
            counter = 0;
          }
        }

        cudaSetDevice(0);
        total_frames += nDev * nnet_in[0].NumRows();
    }
  	if(nnet_in != NULL){
  		free(nnet_in);
  		nnet_in = NULL;
  	}
  	if(nnet_out != NULL){
  		free(nnet_out);
  		nnet_out = NULL;
  	}
  	if(obj_diff != NULL){
  		free(obj_diff);
  		obj_diff = NULL;
  	}
  	if(nnet_tgt != NULL){
  		free(nnet_tgt);
  		nnet_tgt = NULL;
  	}
  	if(example != NULL){
  		free(example);
  		example = NULL;
  	}

  }

  ~DoBackpropParallelClass() {

  }

 private:
  const Nnet &nnet_;
  ExamplesRepository *repository_;
  std::string objective_function_;
  double *tot_weight_ptr_;
  double *log_prob_ptr_;
  double tot_weight_;
  double log_prob_; // log-like times num frames.
  bool crossvalidate_;
};

void DoBackpropParallel(const Nnet &nnet,
						  Nnet &nnet_transf,
                          SequentialBaseFloatMatrixReader &feature_reader,
                          RandomAccessPosteriorReader &targets_reader,
                          std::string &objective_function,
                          std::string &frame_weights,
                          std::string &utt_weights,
                          NnetDataRandomizerOptions &rnd_opts,
                          bool crossvalidate) {


  	ExamplesRepository repository; // handles parallel programming issues regarding
  	// the "examples" of data.
  	double tot_log_prob = 0.0;
  	double tot_weight = 0.0;
  	int32 num_threads = 1;
  	kaldi::int64 total_frames = 0;
  	int32 num_other_error = 0;
  	int32 num_no_tgt_mat = 0;
  	int32 num_done = 0 ;
  	int32 length_tolerance = 5 ;
  	CuMatrix<BaseFloat> feats_transf;
  	Timer time;

  	DoBackpropParallelClass c(nnet, &repository, objective_function, &tot_weight, &tot_log_prob, crossvalidate);

  
    // The initialization of the following class spawns the threads that
    // process the examples.  They get re-joined in its destructor.
    MultiThreader<DoBackpropParallelClass> m(num_threads, c);

    NnetExample examples;
    RandomizerMask randomizer_mask(rnd_opts);
    MatrixRandomizer feature_randomizer(rnd_opts);
    PosteriorRandomizer targets_randomizer(rnd_opts);
    VectorRandomizer weights_randomizer(rnd_opts);
    RandomAccessBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      weights_reader.Open(frame_weights);
    }

    RandomAccessBaseFloatReader utt_weights_reader;
    if (utt_weights != "") {
      utt_weights_reader.Open(utt_weights);
    }

    while (!feature_reader.Done()) {

    	for ( ; !feature_reader.Done(); feature_reader.Next()) {

    		if (feature_randomizer.IsFull()) {
          		// break the loop without calling Next(),
          		// we keep the 'utt' for next round,
    			break;
    		}
    		std::string utt = feature_reader.Key();
    		KALDI_VLOG(3) << "Reading " << utt;
        	// check that we have targets,
    		if (!targets_reader.HasKey(utt)) {
    			KALDI_WARN << utt << ", missing targets";
    			num_no_tgt_mat++;
    			continue;
    		}
        	// check we have per-frame weights,
    		if (frame_weights != "" && !weights_reader.HasKey(utt)) {
    			KALDI_WARN << utt << ", missing per-frame weights";
    			num_other_error++;
    			continue;
    		}
        	// check we have per-utterance weights,
    		if (utt_weights != "" && !utt_weights_reader.HasKey(utt)) {
    			KALDI_WARN << utt << ", missing per-utterance weight";
    			num_other_error++;
    			continue;
    		}
        	// get feature / target pair,
    		Matrix<BaseFloat> mat = feature_reader.Value();
    		Posterior targets = targets_reader.Value(utt);
        	// get per-frame weights,
    		Vector<BaseFloat> weights;
    		if (frame_weights != "") {
    			weights = weights_reader.Value(utt);
        	} else {  // all per-frame weights are 1.0,
        		weights.Resize(mat.NumRows());
        		weights.Set(1.0);
        	}
        	// multiply with per-utterance weight,
        	if (utt_weights != "") {
        		BaseFloat w = utt_weights_reader.Value(utt);
        		KALDI_ASSERT(w >= 0.0);
          		if (w == 0.0) continue;  // remove sentence from training,
          		weights.Scale(w);
          	}

        	// correct small length mismatch or drop sentence,
          	{
          	// add lengths to vector,
          		std::vector<int32> length;
          		length.push_back(mat.NumRows());
          		length.push_back(targets.size());
          		length.push_back(weights.Dim());
          	// find min, max,
          		int32 min = *std::min_element(length.begin(), length.end());
          		int32 max = *std::max_element(length.begin(), length.end());
          	// fix or drop ?
          		if (max - min < length_tolerance) {
            // we truncate to shortest,
          			if (mat.NumRows() != min) mat.Resize(min, mat.NumCols(), kCopyData);
          			if (targets.size() != min) targets.resize(min);
          			if (weights.Dim() != min) weights.Resize(min, kCopyData);
          		} else {
          			KALDI_WARN << "Length mismatch! Targets " << targets.size()
          			<< ", features " << mat.NumRows() << ", " << utt;
          			num_other_error++;
          			continue;
          		}
          	}
        	// apply feature transform (if empty, input is copied),
          	nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);

        	// remove frames with '0' weight from training,
          	{
          	// are there any frames to be removed? (frames with zero weight),
          		BaseFloat weight_min = weights.Min();
          		KALDI_ASSERT(weight_min >= 0.0);
          		if (weight_min == 0.0) {
            // create vector with frame-indices to keep,
          			std::vector<MatrixIndexT> keep_frames;
          			for (int32 i = 0; i < weights.Dim(); i++) {
          				if (weights(i) > 0.0) {
          					keep_frames.push_back(i);
          				}
          			}
            		// when all frames are removed, we skip the sentence,
          			if (keep_frames.size() == 0) continue;

            		// filter feature-frames,
          			CuMatrix<BaseFloat> tmp_feats(keep_frames.size(), feats_transf.NumCols());
          			tmp_feats.CopyRows(feats_transf, CuArray<MatrixIndexT>(keep_frames));
          			tmp_feats.Swap(&feats_transf);

            		// filter targets,
          			Posterior tmp_targets;
          			for (int32 i = 0; i < keep_frames.size(); i++) {
          				tmp_targets.push_back(targets[keep_frames[i]]);
          			}
          			tmp_targets.swap(targets);

            		// filter weights,
          			Vector<BaseFloat> tmp_weights(keep_frames.size());
          			for (int32 i = 0; i < keep_frames.size(); i++) {
          				tmp_weights(i) = weights(keep_frames[i]);
          			}
          			tmp_weights.Swap(&weights);
          		}
          	}

        	// pass data to randomizers,
          	KALDI_ASSERT(feats_transf.NumRows() == targets.size());
          	feature_randomizer.AddData(feats_transf);
          	targets_randomizer.AddData(targets);
          	weights_randomizer.AddData(weights);
          	num_done++;

        	// report the speed,
          	if (num_done % 5000 == 0) {
          		double time_now = time.Elapsed();
          		KALDI_VLOG(1) << "After " << num_done << " utterances: "
          		<< "time elapsed = " << time_now / 60 << " min; "
          		<< "processed " << total_frames / time_now << " frames per sec.";
          	}

        }

    	// randomize,
    	if (!crossvalidate) {
        	const std::vector<int32>& mask =
          		randomizer_mask.Generate(feature_randomizer.NumFrames());
        	feature_randomizer.Randomize(mask);
        	targets_randomizer.Randomize(mask);
        	weights_randomizer.Randomize(mask);
    	}
    	// train with data from randomizers (using mini-batches),
    	for ( ; !feature_randomizer.Done(); feature_randomizer.Next(),
                                        	targets_randomizer.Next(),
                                        	weights_randomizer.Next()) {
    		NnetExample example(&feature_randomizer.Value(), &targets_randomizer.Value(), &weights_randomizer.Value());
    		repository.AcceptExamples(example);
    	}

    }

  
  	KALDI_LOG << "Did backprop on " << tot_weight << " examples, average log-prob "
              << "per frame is " << (tot_log_prob / tot_weight);
    KALDI_LOG << "[this line is to be parsed by a script:] log-prob-per-frame="
              << (tot_log_prob / tot_weight);
    //return tot_log_prob;
}


}//end of namespace nnet
}//end of namespace kaldi