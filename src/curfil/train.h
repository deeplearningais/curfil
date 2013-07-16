#ifndef CURFIL_TRAIN_H
#define CURFIL_TRAIN_H

#include "random_forest_image.h"

namespace curfil
{

void determineImageCacheSizeAndSamplesPerBatch(const std::vector<LabeledRGBDImage>& images,
        const std::vector<int>& deviceId, const size_t featureCount, const size_t numThresholds,
        size_t imageCacheSizeMB, unsigned int& imageCacheSize, unsigned int& maxSamplesPerBatch);

RandomForestImage train(std::vector<LabeledRGBDImage>& image, size_t trees,
        const TrainingConfiguration& configuration, bool trainTreesInParallel);

}

#endif
