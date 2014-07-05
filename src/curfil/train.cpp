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
#include "train.h"

#include <cuda.h>
#include <iomanip>

#include "image.h"
#include "random_forest_image.h"
#include "random_tree_image.h"
#include "utils.h"

namespace curfil
{

void determineImageCacheSizeAndSamplesPerBatch(const std::vector<LabeledRGBDImage>& images,
        const std::vector<int>& deviceIds, const size_t featureCount, const size_t numThresholds,
        size_t imageCacheSizeMB, unsigned int& imageCacheSize, unsigned int& maxSamplesPerBatch) {

    if (deviceIds.empty()) {
        throw std::runtime_error("got no device IDs");
    }

    if (deviceIds.size() != 1) {
        throw std::runtime_error("only one device ID supported currently");
    }

    size_t freeMemoryOnGPU = utils::getFreeMemoryOnGPU(deviceIds[0]);

    if (imageCacheSizeMB == 0) {
        imageCacheSizeMB = freeMemoryOnGPU * 0.66 / 1024 / 1024;
    }

    // very defensive estimate to avoid out of memory errors
    long remainingMemoryOnGPU = (freeMemoryOnGPU - imageCacheSizeMB * 1024lu * 1024lu) / 3;
    // size for histogram counters
    remainingMemoryOnGPU -= 10 * (2 * sizeof(WeightType) * featureCount * numThresholds);
    size_t sizePerSample = 2 * sizeof(FeatureResponseType) * featureCount;

    maxSamplesPerBatch = remainingMemoryOnGPU / sizePerSample;
    maxSamplesPerBatch = std::min(maxSamplesPerBatch, 50000u);

    if (maxSamplesPerBatch < 1000) {
        throw std::runtime_error("memory headroom on GPU too low. try to decrease image cache size manually");
    }

    CURFIL_INFO("max samples per batch: " << maxSamplesPerBatch);

    if (images.size() * images[0].getSizeInMemory() <= imageCacheSizeMB * 1024lu * 1024lu) {
        imageCacheSize = images.size();
    } else {
        imageCacheSize = imageCacheSizeMB * 1024lu * 1024lu / images[0].getSizeInMemory();
    }

    CURFIL_INFO((boost::format("image cache size: %d images (%.1f MB)")
            % imageCacheSize
            % (imageCacheSize * images[0].getSizeInMemory() / 1024.0 / 1024.0)).str());

    if (imageCacheSizeMB * 1024lu * 1024lu >= freeMemoryOnGPU) {
        throw std::runtime_error("image cache size too large");
    }
}

RandomForestImage train(std::vector<LabeledRGBDImage>& images, size_t trees,
        const TrainingConfiguration& configuration, size_t numLabels, bool trainTreesInParallel) {

    CURFIL_INFO("trees: " << trees);
    CURFIL_INFO("training trees in parallel: " << trainTreesInParallel);
    CURFIL_INFO(configuration);

    // Train

    RandomForestImage randomForest(trees, configuration);

    utils::Timer trainTimer;
    randomForest.train(images, numLabels, !trainTreesInParallel);
    trainTimer.stop();

    CURFIL_INFO("training took " << trainTimer.format(2) <<
            " (" << std::setprecision(3) << trainTimer.getSeconds() / 60.0 << " min)");

/*      // This feature is only experimental, it should be rewritten in a much faster way, and a CPU version should be added
        bool onGPU = randomForest.getConfiguration().getAccelerationMode() == GPU_ONLY;
        bool useDepthImages = randomForest.getConfiguration().isUseDepthImages();

        size_t grainSize = 1;
        if (!onGPU) {
            grainSize = images.size();
        }

        //should get the correct histogram bias
        randomForest.normalizeHistograms(0.0);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, images.size(), grainSize),
                [&](const tbb::blocked_range<size_t>& range) {
                    for(size_t imageNr = range.begin(); imageNr != range.end(); imageNr++) {
                        const LabeledRGBDImage imagePair = images[imageNr];
                        const RGBDImage& trainingImage = imagePair.getRGBDImage();
                        const LabelImage& groundTruth = imagePair.getLabelImage();
                        LabelImage prediction(trainingImage.getWidth(), trainingImage.getHeight());
                        prediction = randomForest.improveHistograms(trainingImage, groundTruth, true, useDepthImages);
                    }
        });
        randomForest.updateTreesHistograms();
*/
    std::cout << randomForest;
    for (const auto& featureCount : randomForest.countFeatures()) {
        const std::string featureType = featureCount.first;
        CURFIL_INFO("feature " << featureType << ": " << featureCount.second);
    }

    return randomForest;
}

}
