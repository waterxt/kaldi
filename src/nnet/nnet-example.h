// nnet/nnet-example.h
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

#ifndef KALDI_NNET_NNET_EXAMPLE_H_
#define KALDI_NNET_NNET_EXAMPLE_H_

#include "util/table-types.h"
#include "lat/kaldi-lattice.h"
#include "util/kaldi-semaphore.h"
#include "nnet/nnet-loss.h"


namespace kaldi{
namespace nnet1{

struct NnetExample {

	 const CuMatrixBase<BaseFloat> *mat ;
	 const Posterior *targets ;
   const Vector<BaseFloat> *frame_weights;

	NnetExample() { }

	NnetExample(const CuMatrixBase<BaseFloat> *features, const Posterior *labels, const Vector<BaseFloat> *weight);

};


class ExamplesRepository {
 public:
  /// The following function is called by the code that reads in the examples,
  /// with a batch of examples.  [It will empty the vector "examples").
  void AcceptExamples(NnetExample &example);

  /// The following function is called by the code that reads in the examples,
  /// when we're done reading examples.
  void ExamplesDone();
  
  /// This function is called by the code that does the training.  It gets the
  /// training examples, and if they are available, puts them in "examples" and
  /// returns true.  It returns false when there are no examples left and
  /// ExamplesDone() has been called.
  bool ProvideExamples(NnetExample &example);
  
  ExamplesRepository(): empty_semaphore_(1), done_(false) { }
 private:
  
  Semaphore full_semaphore_;
  Semaphore empty_semaphore_;

  std::vector<NnetExample> examples_;
  bool done_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(ExamplesRepository);

};

}

}














#endif