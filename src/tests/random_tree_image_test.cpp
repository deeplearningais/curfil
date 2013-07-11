#define BOOST_TEST_MODULE example

#include <boost/filesystem.hpp>
#include <boost/test/included/unit_test.hpp>
#include <math.h>
#include <stdlib.h>
#include <tbb/task_scheduler_init.h>
#include <vector>
#include <vigra/colorconversions.hxx>
#include <vigra/copyimage.hxx>
#include <vigra/impex.hxx>
#include <vigra/stdimage.hxx>
#include <vigra/transformimage.hxx>

#include "export.h"
#include "image.h"
#include "predict.h"
#include "random_forest_image.h"
#include "random_tree_image.h"

using namespace curfil;

static const int NUM_THREADS = 4;
static const std::string folderOutput("test.out");

BOOST_AUTO_TEST_SUITE(RandomTreeImageTest)

static std::string getFolderTraining() {
    if (boost::unit_test::framework::master_test_suite().argc < 2) {
        throw std::runtime_error("please specify folder with testdata");
    }
    return boost::unit_test::framework::master_test_suite().argv[1];
}

static double predict(RandomForestImage& randomForest) {

    const bool useCIELab = true;
    const bool useDepthFilling = false;
    const auto testing = loadImagePair(getFolderTraining() + "/testing1_colors.png", useCIELab, useDepthFilling);
    const LabelImage& groundTruth = testing.getLabelImage();

    randomForest.normalizeHistograms(0.0);

    LabelImage prediction = randomForest.predict(testing.getRGBDImage());

    boost::filesystem::create_directory(folderOutput);

    groundTruth.save(folderOutput + "/testing1_groundTruth.png");
    prediction.save(folderOutput + "/testing1_prediction_singletree_cpu_gpu.png");

    ConfusionMatrix confusionMatrix;
    double accuracy = 100 * calculatePixelAccuracy(prediction, groundTruth, true, &confusionMatrix);
    double accuracyWithoutVoid = 100 * calculatePixelAccuracy(prediction, groundTruth, false);

    CURFIL_INFO("accuracy (no void): " << accuracy << " (" << accuracyWithoutVoid << ")");

    return accuracy;
}

BOOST_AUTO_TEST_CASE(trainTest) {
    const bool useCIELab = true;
    const bool useDepthFilling = false;

    std::vector<LabeledRGBDImage> trainImages;
    trainImages.push_back(loadImagePair(getFolderTraining() + "/training1_colors.png", useCIELab, useDepthFilling));
    trainImages.push_back(loadImagePair(getFolderTraining() + "/training2_colors.png", useCIELab, useDepthFilling));
    trainImages.push_back(loadImagePair(getFolderTraining() + "/training3_colors.png", useCIELab, useDepthFilling));

    tbb::task_scheduler_init init(NUM_THREADS);

    // Train

    unsigned int samplesPerImage = 500;
    unsigned int featureCount = 500;
    unsigned int minSampleCount = 100;
    int maxDepth = 10;
    uint16_t boxRadius = 127;
    uint16_t regionSize = 16;
    uint16_t thresholds = 50;
    int numThreads = NUM_THREADS;
    int maxImages = 10;
    int imageCacheSize = 10;
    unsigned int maxSamplesPerBatch = 5000;
    AccelerationMode accelerationMode = AccelerationMode::GPU_AND_CPU_COMPARE;

    const int SEED = 4713;

    TrainingConfiguration configuration(SEED, samplesPerImage, featureCount, minSampleCount, maxDepth, boxRadius,
            regionSize, thresholds, numThreads, maxImages, imageCacheSize, maxSamplesPerBatch, accelerationMode);

    RandomForestImage randomForest(1, configuration);
    randomForest.train(trainImages);

    double accuracy = predict(randomForest);

    BOOST_CHECK_CLOSE_FRACTION(73, accuracy, 10.0);
}

BOOST_AUTO_TEST_CASE(trainTestGPU) {
    const bool useCIELab = true;
    const bool useDepthFilling = false;

    std::vector<LabeledRGBDImage> trainImages;
    trainImages.push_back(loadImagePair(getFolderTraining() + "/training1_colors.png", useCIELab, useDepthFilling));
    trainImages.push_back(loadImagePair(getFolderTraining() + "/training2_colors.png", useCIELab, useDepthFilling));
    trainImages.push_back(loadImagePair(getFolderTraining() + "/training3_colors.png", useCIELab, useDepthFilling));

    tbb::task_scheduler_init init(NUM_THREADS);

    // Train

    unsigned int samplesPerImage = 500;
    unsigned int featureCount = 500;
    unsigned int minSampleCount = 100;
    int maxDepth = 10;
    uint16_t boxRadius = 127;
    uint16_t regionSize = 16;
    uint16_t thresholds = 50;
    int maxImages = 10;
    int imageCacheSize = 10;
    unsigned int maxSamplesPerBatch = 5000;
    AccelerationMode accelerationMode = AccelerationMode::GPU_ONLY;

    const int SEED = 4711;

    TrainingConfiguration configuration(SEED, samplesPerImage, featureCount, minSampleCount, maxDepth, boxRadius,
            regionSize, thresholds, NUM_THREADS, maxImages, imageCacheSize, maxSamplesPerBatch, accelerationMode);

    RandomForestImage randomForest(1, configuration);
    randomForest.train(trainImages);

    double accuracy = predict(randomForest);

    BOOST_CHECK_CLOSE_FRACTION(73, accuracy, 10.0);
}

BOOST_AUTO_TEST_CASE(trainTestEnsemble) {

    const bool useCIELab = true;
    const bool useDepthFilling = false;

    std::vector<LabeledRGBDImage> trainImages;
    trainImages.push_back(loadImagePair(getFolderTraining() + "/training1_colors.png", useCIELab, useDepthFilling));
    trainImages.push_back(loadImagePair(getFolderTraining() + "/training2_colors.png", useCIELab, useDepthFilling));
    trainImages.push_back(loadImagePair(getFolderTraining() + "/training3_colors.png", useCIELab, useDepthFilling));

    // Train

    unsigned int samplesPerImage = 2000;
    unsigned int featureCount = 500;
    unsigned int minSampleCount = 32;
    int maxDepth = 14;
    uint16_t boxRadius = 120;
    uint16_t regionSize = 10;
    uint16_t thresholds = 50;
    int maxImages = 3;
    int imageCacheSize = 3;
    unsigned int maxSamplesPerBatch = 5000;
    AccelerationMode accelerationMode = AccelerationMode::GPU_AND_CPU_COMPARE;

    const int SEED = 64684263;

    TrainingConfiguration configuration(SEED, samplesPerImage, featureCount, minSampleCount, maxDepth, boxRadius,
            regionSize, thresholds, NUM_THREADS, maxImages, imageCacheSize, maxSamplesPerBatch, accelerationMode);

    size_t trees = 3;

    RandomForestImage randomForest(trees, configuration);
    randomForest.train(trainImages);

    double accuracy = predict(randomForest);

    BOOST_CHECK_CLOSE_FRACTION(76, accuracy, 5.0);

    // 2nd attempt on GPU
    accelerationMode = AccelerationMode::GPU_ONLY;
    double accuracy2;
    {
        TrainingConfiguration configuration(SEED, samplesPerImage, featureCount, minSampleCount, maxDepth, boxRadius,
                regionSize, thresholds, NUM_THREADS, maxImages, imageCacheSize, maxSamplesPerBatch, accelerationMode);
        RandomForestImage randomForest(trees, configuration);
        randomForest.train(trainImages);

        accuracy2 = predict(randomForest);

        BOOST_CHECK_CLOSE_FRACTION(accuracy, accuracy2, 2.5);
    }

    // 3rd attempt on GPU
    {
        TrainingConfiguration configuration(SEED, samplesPerImage, featureCount, minSampleCount, maxDepth, boxRadius,
                regionSize, thresholds, NUM_THREADS, maxImages, imageCacheSize, maxSamplesPerBatch, accelerationMode);
        RandomForestImage randomForest(trees, configuration);
        randomForest.train(trainImages);

        double accuracy3 = predict(randomForest);

        BOOST_CHECK_CLOSE_FRACTION(accuracy2, accuracy3, 1.0);
    }

}
BOOST_AUTO_TEST_SUITE_END()
