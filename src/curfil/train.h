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
        const TrainingConfiguration& configuration, bool trainTreesInParallel);

}

#endif
