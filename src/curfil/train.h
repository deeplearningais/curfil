#if 0
#######################################################################################
# The MIT License

# Copyright (c) 2014       Hannes Schulz, University of Bonn  <schulz@ais.uni-bonn.de>
# Copyright (c) 2013       Benedikt Waldvogel, University of Bonn <mail@bwaldvogel.de>
# Copyright (c) 2008-2009  Sebastian Nowozin                       <nowozin@gmail.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#######################################################################################
#endif
#ifndef CURFIL_TRAIN_H
#define CURFIL_TRAIN_H

#include "random_forest_image.h"

namespace curfil
{

/**
 * Helper function to automatically determine the best configuration for the image cache and the number of samples (pixels) per batch, depending on the available GPU.
 */
void determineImageCacheSizeAndSamplesPerBatch(const std::vector<LabeledRGBDImage>& images,
        const std::vector<int>& deviceId, const size_t featureCount, const size_t numThresholds,
        size_t imageCacheSizeMB, unsigned int& imageCacheSize, unsigned int& maxSamplesPerBatch);

/**
 * Train a random forest with the given training images and configuration.
 *
 * @param images list of training images
 * @param trees number of trees to train
 * @param configuration the training configuration
 * @param trainTreeInParallel whether to train each tree in sequential order or in parallel.
 */
RandomForestImage train(std::vector<LabeledRGBDImage>& images, size_t trees,
        const TrainingConfiguration& configuration, size_t numLabels, bool trainTreesInParallel);

}

#endif
