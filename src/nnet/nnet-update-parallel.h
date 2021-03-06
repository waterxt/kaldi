// nnet/nnet-update-parallel.h
// CopyRight
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
#include "base/kaldi-common.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "util/common-utils.h"
#include "nnet/nnet-trnopts.h"

namespace kaldi{
namespace nnet1{

void DoBackpropParallel(const Nnet &nnet,
						Nnet &nnet_transf,
                        SequentialBaseFloatMatrixReader &features,
                        RandomAccessPosteriorReader &targets,
                        std::string &objective_function,
                        std::string &frame_weights,
                        std::string &utt_weights,
                        NnetDataRandomizerOptions &rnd_opts,
                        bool crossvalidate);

}
}