// nnetbin/nnet-train-frmshuff-single-thread.cc

// Copyright 2017 (tao xu)

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

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-update-parallel.h"

int main(int argc, char *argv[]){
	using namespace kaldi;
	using namespace kaldi::nnet1;
	typedef kaldi::int32 int32;
	try{
	const char *usage =
      "Perform one iteration (epoch) of Neural Network training with\n"
      "mini-batch Stochastic Gradient Descent. The training targets\n"
      "are usually pdf-posteriors, prepared by ali-to-post.\n"
      "Usage:  nnet-train-frmshuff-single-thread [options] <feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n"
      "e.g.: nnet-train-frmshuff-single-thread scp:feats.scp ark:posterior.ark nnet.init nnet.iter1\n";
    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);
    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);

    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");

    bool crossvalidate = false;
    po.Register("cross-validate", &crossvalidate,
        "Perform cross-validation (don't back-propagate)");

    bool randomize = true;
    po.Register("randomize", &randomize,
        "Perform the frame-level shuffling within the Cache::");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
        "Feature transform in Nnet format");

    std::string objective_function = "xent";
    po.Register("objective-function", &objective_function,
        "Objective function : xent|mse|multitask");

    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance,
        "Allowed length mismatch of features/targets/weights "
        "(in frames, we truncate to the shortest)");

    std::string frame_weights;
    po.Register("frame-weights", &frame_weights,
        "Per-frame weights, used to re-scale gradients.");

    std::string utt_weights;
    po.Register("utt-weights", &utt_weights,
        "Per-utterance weights, used to re-scale frame-weights.");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu,
        "yes|no|optional, only has effect if compiled with CUDA");

    po.Read(argc, argv);

    if (po.NumArgs() != 3 + (crossvalidate ? 0 : 1)) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
      targets_rspecifier = po.GetArg(2),
      model_filename = po.GetArg(3);

    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(4);
    }

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;
    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    nnet.SetTrainOptions(trn_opts);

    if (crossvalidate) {
      nnet_transf.SetDropoutRate(0.0);
      nnet.SetDropoutRate(0.0);
    }

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader targets_reader(targets_rspecifier);
    RandomAccessBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      weights_reader.Open(frame_weights);
    }
    RandomAccessBaseFloatReader utt_weights_reader;
    if (utt_weights != "") {
      utt_weights_reader.Open(utt_weights);
    }


	DoBackpropParallel(nnet,
					   nnet_transf,
                       feature_reader,
                       targets_reader,
                       objective_function,
                       frame_weights,
                       utt_weights,
                       rnd_opts,
                       crossvalidate);

	return 0;
	} catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }






}