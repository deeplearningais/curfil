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

    imageCacheSizeMB = 0;
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
        const TrainingConfiguration& configuration, bool trainTreesInParallel) {

    CURFIL_INFO("trees: " << trees);
    CURFIL_INFO("training trees in parallel: " << trainTreesInParallel);
    CURFIL_INFO(configuration);

    // Train

    RandomForestImage randomForest(trees, configuration);

    utils::Timer trainTimer;
    randomForest.train(images, !trainTreesInParallel);
    trainTimer.stop();

    CURFIL_INFO("training took " << trainTimer.format(2) <<
            " (" << std::setprecision(3) << trainTimer.getSeconds() / 60.0 << " min)");

    std::cout << randomForest;
    for (const auto& featureCount : randomForest.countFeatures()) {
        const std::string featureType = featureCount.first;
        CURFIL_INFO("feature " << featureType << ": " << featureCount.second);
    }

    return randomForest;
}

}
