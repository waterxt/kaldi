//nnet/nnet-example.cc
//CopyRight: tao xu

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

#include "nnet/nnet-example.h"

namespace kaldi{
namespace nnet1{

NnetExample::NnetExample(Matrix<BaseFloat> &features, Posterior &labels){

	mat.Swap(features) ;
	targets.Swap(labels);
}

void ExamplesRepository::AcceptExamples(NnetExample &example){

  KALDI_ASSERT(!examples->empty());
  empty_semaphore_.Wait();
  KALDI_ASSERT(examples_.empty());
  examples_.push_back(example);
  full_semaphore_.Signal();

}

void ExamplesRepository::ExamplesDone(){

  empty_semaphore_.Wait();
  KALDI_ASSERT(examples_.empty());
  done_ = true;
  full_semaphore_.Signal();
}

bool ExamplesRepository::ProvideExamples(NnetExample &example){
  full_semaphore_.Wait();
  if (done_) {
    KALDI_ASSERT(examples_.empty());
    full_semaphore_.Signal(); // Increment the semaphore so
    // the call by the next thread will not block.
    return false; // no examples to return-- all finished.
  } else {
    KALDI_ASSERT(!examples_.empty() && examples->empty());
    examples_->pop_back(example);
    empty_semaphore_.Signal();
    return true;
  }
}

}

}