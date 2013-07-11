#define BOOST_TEST_MODULE example

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/format.hpp>
#include <boost/functional/hash.hpp>
#include <boost/make_shared.hpp>
#include <boost/test/included/unit_test.hpp>
#include <vector>

#include "random_tree_image_gpu.h"
#include "random_tree_image.h"
#include "test_common.h"
#include "utils.h"

BOOST_AUTO_TEST_SUITE(feature_generation)

using namespace curfil;

static const int SEED = 4711;
static const int NUM_THREADS = 1;
static const uint16_t BOX_RADIUS = 127;

template<class T>
static void updateHash(const boost::hash<size_t>& hasher, size_t& hash, const T& value) {
    // extracted from boost headers
    hash ^= hasher(value) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
}

template<class T>
static void updateHash(const boost::hash<size_t>& hasher, size_t& hash, const std::vector<T>& values) {
    for (size_t i = 0; i < values.size(); i++) {
        updateHash(hasher, hash, values[i]);
    }
}

template<class T>
static std::string meanStddevMinMax(const std::vector<T>& values, double& min, double& max, double& mean,
        double& stddev) {
    boost::accumulators::accumulator_set<T,
            boost::accumulators::features<
                    boost::accumulators::tag::min,
                    boost::accumulators::tag::max,
                    boost::accumulators::tag::mean,
                    boost::accumulators::tag::variance> > acc;

    acc = std::for_each(values.begin(), values.end(), acc);

    min = boost::accumulators::min(acc);
    max = boost::accumulators::max(acc);
    stddev = std::sqrt(boost::accumulators::variance(acc));
    mean = boost::accumulators::mean(acc);

    return boost::str(boost::format("mean: %lf, stddev: %lf, min: %lf, max: %lf")
            % mean
            % stddev
            % min
            % max);
}

template<class T>
static void checkMeanStddevMinMax(const std::string key, const std::vector<T>& values,
        double mean,
        double stddev,
        double min,
        double max,
        double meanTolerance,
        double stddevTolerance,
        double minTolerance,
        double maxTolerance) {

    double actualMin, actualMax, actualMean, actualStddev;
    const std::string msg = meanStddevMinMax(values, actualMin, actualMax, actualMean, actualStddev);
    CURFIL_INFO(key << ": " << msg);

    BOOST_CHECK_LE(abs(mean - actualMean), meanTolerance);
    BOOST_CHECK_LE(abs(stddev - actualStddev), stddevTolerance);
    BOOST_CHECK_LE(abs(min - actualMin), minTolerance);
    BOOST_CHECK_LE(abs(max - actualMax), maxTolerance);
}

static void assertLessThanEqual(const ImageFeatureFunction& feature1, const ImageFeatureFunction& feature2) {

    int key1 = feature1.getSortKey();
    int key2 = feature2.getSortKey();

    if (key1 > key2) {
        BOOST_CHECK(false);
    }

    BOOST_CHECK_LE(key1, key2);

    if (feature1.getType() != feature2.getType()) {
        BOOST_CHECK_LT(feature1.getType(), feature2.getType());
        return;
    }

    if (feature1.getChannel1() != feature2.getChannel1()) {
        BOOST_CHECK_LT(feature1.getChannel1(), feature2.getChannel1());
        return;
    }

    if (feature1.getChannel2() != feature2.getChannel2()) {
        BOOST_CHECK_LT(feature1.getChannel2(), feature2.getChannel2());
        return;
    }

    if (feature1.getOffset1().getY() != feature2.getOffset1().getY()) {
        BOOST_CHECK_LT(feature1.getOffset1().getY(), feature2.getOffset1().getY());
        return;
    }

    if (feature1.getOffset1().getX() != feature2.getOffset1().getX()) {
        BOOST_CHECK_LT(feature1.getOffset1().getX(), feature2.getOffset1().getX());
        return;
    }
}

static void assertSorted(const ImageFeaturesAndThresholds<cuv::host_memory_space>& features) {

    const size_t numFeatures = features.m_features.shape(1);

    for (size_t feat = 1; feat < numFeatures; feat++) {
        const ImageFeatureFunction feature1 = features.getFeatureFunction(feat - 1);
        const ImageFeatureFunction feature2 = features.getFeatureFunction(feat);

        assertLessThanEqual(feature1, feature2);
    }
}

static void assertEqual(ImageFeaturesAndThresholds<cuv::host_memory_space>& features1,
        ImageFeaturesAndThresholds<cuv::host_memory_space>& features2, TrainingConfiguration& configuration) {

    BOOST_REQUIRE_EQUAL(static_cast<int>(features1.m_features.ndim()),
            static_cast<int>(features2.m_features.ndim()));
    BOOST_REQUIRE_EQUAL(static_cast<int>(features1.m_thresholds.ndim()),
            static_cast<int>(features2.m_thresholds.ndim()));

    for (int dim = 0; dim < features1.m_features.ndim(); dim++) {
        BOOST_CHECK_EQUAL(static_cast<int>(features1.m_features.shape(dim)),
                static_cast<int>(features2.m_features.shape(dim)));
    }

    for (int dim = 0; dim < features1.m_thresholds.ndim(); dim++) {
        BOOST_CHECK_EQUAL(static_cast<int>(features1.m_thresholds.shape(dim)),
                static_cast<int>(features2.m_thresholds.shape(dim)));
    }

    const size_t numFeatures = features1.m_features.shape(1);
    BOOST_REQUIRE_EQUAL(numFeatures, configuration.getFeatureCount());
    BOOST_REQUIRE_EQUAL(static_cast<unsigned int>(features1.thresholds().shape(1)), configuration.getFeatureCount());

    BOOST_CHECK(features1.m_features == features2.m_features);
    BOOST_CHECK(features1.m_thresholds == features2.m_thresholds);
}

static size_t analyzeFeatures(const TrainingConfiguration& configuration,
        const ImageFeaturesAndThresholds<cuv::host_memory_space>& features) {

    assertSorted(features);

    int boxRadius = configuration.getBoxRadius();
    int regionSize = configuration.getRegionSize();

    BOOST_CHECK_EQUAL(2, static_cast<int>(features.thresholds().ndim()));
    BOOST_CHECK_EQUAL(configuration.getThresholds(), static_cast<uint16_t>(features.thresholds().shape(0)));
    BOOST_CHECK_EQUAL(configuration.getFeatureCount(), static_cast<unsigned int>(features.thresholds().shape(1)));

    std::vector<int> featureTypes;
    std::vector<int> channels1;
    std::vector<int> channels2;
    std::vector<int> offsets1X;
    std::vector<int> offsets1Y;
    std::vector<int> offsets2X;
    std::vector<int> offsets2Y;
    std::vector<int> regions1X;
    std::vector<int> regions1Y;
    std::vector<int> regions2X;
    std::vector<int> regions2Y;
    std::vector<float> thresholds;

    for (size_t feat = 0; feat < configuration.getFeatureCount(); feat++) {
        const ImageFeatureFunction& feature = features.getFeatureFunction(feat);

        BOOST_CHECK(feature.getType() == COLOR || feature.getType() == DEPTH);

        featureTypes.push_back(feature.getType());

        BOOST_CHECK(feature.getChannel1() >= 0 && feature.getChannel1() < 3);
        BOOST_CHECK(feature.getChannel2() >= 0 && feature.getChannel2() < 3);

        channels1.push_back(feature.getChannel1());
        channels2.push_back(feature.getChannel2());

        offsets1X.push_back(feature.getOffset1().getX());
        offsets1Y.push_back(feature.getOffset1().getY());
        offsets2X.push_back(feature.getOffset2().getX());
        offsets2Y.push_back(feature.getOffset2().getY());

        BOOST_CHECK(feature.getOffset1().getX() >= -boxRadius && feature.getOffset1().getX() <= boxRadius);
        BOOST_CHECK(feature.getOffset1().getY() >= -boxRadius && feature.getOffset1().getY() <= boxRadius);
        BOOST_CHECK(feature.getOffset2().getX() >= -boxRadius && feature.getOffset2().getX() <= boxRadius);
        BOOST_CHECK(feature.getOffset2().getY() >= -boxRadius && feature.getOffset2().getY() <= boxRadius);

        regions1X.push_back(feature.getRegion1().getX());
        regions1Y.push_back(feature.getRegion1().getY());
        regions2X.push_back(feature.getRegion2().getX());
        regions2Y.push_back(feature.getRegion2().getY());

        BOOST_CHECK(feature.getRegion1().getX() > 0 && feature.getRegion1().getX() <= regionSize);
        BOOST_CHECK(feature.getRegion1().getY() > 0 && feature.getRegion1().getY() <= regionSize);
        BOOST_CHECK(feature.getRegion2().getX() > 0 && feature.getRegion2().getX() <= regionSize);
        BOOST_CHECK(feature.getRegion2().getY() > 0 && feature.getRegion2().getY() <= regionSize);

        for (uint16_t thresh = 0; thresh < configuration.getThresholds(); thresh++) {
            float v = features.getThreshold(thresh, feat);
            thresholds.push_back(v);
        }
    }

    // 2013-Feb-21 12:43:40.412222  INFO    feature types: mean: 0.4955, stddev: 0.49998, min: 0, max: 1
    // 2013-Feb-21 12:43:40.412289  INFO    channel1: mean: 0.4965, stddev: 0.758939, min: 0, max: 2
    // 2013-Feb-21 12:43:40.412345  INFO    channel2: mean: 0.4955, stddev: 0.76484, min: 0, max: 2
    // 2013-Feb-21 12:43:40.412402  INFO    offset1X: mean: -0.3005, stddev: 73.6856, min: -127, max: 127
    // 2013-Feb-21 12:43:40.412459  INFO    offset1Y: mean: -1.811, stddev: 73.8739, min: -127, max: 127
    // 2013-Feb-21 12:43:40.412515  INFO    offset2X: mean: -0.6175, stddev: 73.6058, min: -127, max: 127
    // 2013-Feb-21 12:43:40.412571  INFO    offset2Y: mean: 1.994, stddev: 73.8874, min: -127, max: 127
    // 2013-Feb-21 12:43:40.412627  INFO    region1X: mean: 8.4045, stddev: 4.70328, min: 1, max: 16
    // 2013-Feb-21 12:43:40.412685  INFO    region1Y: mean: 8.485, stddev: 4.6169, min: 1, max: 16
    // 2013-Feb-21 12:43:40.412741  INFO    region2X: mean: 8.451, stddev: 4.6358, min: 1, max: 16
    // 2013-Feb-21 12:43:40.412803  INFO    region2Y: mean: 8.39, stddev: 4.5642, min: 1, max: 16
    // 2013-Feb-21 12:43:40.414212  INFO    thresholds: mean: 18.7796, stddev: 786.88, min: -5120.38, max: 4951.81

    checkMeanStddevMinMax("featureType", featureTypes, 0.5, 0.5, 0, 1, 0.05, 0.001, 0, 0);

    // mean=0.5 since we have more channel=0 values because of depth feature
    checkMeanStddevMinMax("channel1", channels1, 0.5, 0.75, 0, 2, 0.05, 0.02, 0, 0);
    checkMeanStddevMinMax("channel2", channels2, 0.5, 0.75, 0, 2, 0.05, 0.02, 0, 0);

    checkMeanStddevMinMax("offset1X", offsets1X, 0.0, 75, -127, +127, 5.0, 3.5, 1, 1);
    checkMeanStddevMinMax("offset1Y", offsets1Y, 0.0, 75, -127, +127, 5.0, 3.5, 1, 1);
    checkMeanStddevMinMax("offset2X", offsets2X, 0.0, 75, -127, +127, 5.0, 3.5, 1, 1);
    checkMeanStddevMinMax("offset2Y", offsets2Y, 0.0, 75, -127, +127, 5.0, 3.5, 1, 1);

    checkMeanStddevMinMax("region1X", regions1X, 8.5, 4.6, +1, +16, 0.3, 0.2, 0, 0);
    checkMeanStddevMinMax("region1Y", regions1Y, 8.5, 4.6, +1, +16, 0.3, 0.2, 0, 0);
    checkMeanStddevMinMax("region2X", regions2X, 8.5, 4.6, +1, +16, 0.3, 0.2, 0, 0);
    checkMeanStddevMinMax("region2Y", regions2Y, 8.5, 4.6, +1, +16, 0.3, 0.2, 0, 0);

    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
    CURFIL_INFO("thresholds: " << meanStddevMinMax(thresholds, min, max, mean, stddev));

    BOOST_CHECK_LT(abs(0.0 - mean), 10.0);
    BOOST_CHECK_GT(stddev, 50.0);
    BOOST_CHECK_LT(min, -1000);
    BOOST_CHECK_GT(max, +1000);

    boost::hash<size_t> hasher;
    size_t hash = 0;

    updateHash(hasher, hash, featureTypes);
    updateHash(hasher, hash, channels1);
    updateHash(hasher, hash, channels2);
    updateHash(hasher, hash, offsets1X);
    updateHash(hasher, hash, offsets1Y);
    updateHash(hasher, hash, offsets2X);
    updateHash(hasher, hash, offsets2Y);
    updateHash(hasher, hash, regions1X);
    updateHash(hasher, hash, regions1Y);
    updateHash(hasher, hash, regions2X);
    updateHash(hasher, hash, regions2Y);
    updateHash(hasher, hash, thresholds);

    return hash;
}

BOOST_AUTO_TEST_CASE(testFeatureGenerationCPU) {

    clearImageCache();

    unsigned int samplesPerImage = 500;
    unsigned int featureCount = 500;
    unsigned int minSampleCount = 32;
    int maxDepth = 15;
    uint16_t regionSize = 16;
    static const uint16_t THRESHOLDS = 50;
    int numThreads = NUM_THREADS;
    int maxImages = 10;
    int imageCacheSize = 10;
    unsigned int maxSamplesPerBatch = 5000;
    AccelerationMode accelerationMode = AccelerationMode::CPU_ONLY;

    TrainingConfiguration configuration(SEED, samplesPerImage, featureCount, minSampleCount, maxDepth, BOX_RADIUS,
            regionSize, THRESHOLDS, numThreads, maxImages, imageCacheSize, maxSamplesPerBatch, accelerationMode);

    static const int width = 400;
    static const int height = 400;

    std::vector<PixelInstance> samples;

    RandomSource randomSource(4711);
    Sampler widthSampler = randomSource.uniformSampler(width);
    Sampler heightSampler = randomSource.uniformSampler(height);

    Sampler pixelSampler = randomSource.uniformSampler(1000);
    Sampler depthSampler = randomSource.uniformSampler(1, 90);

    const int NUM_LABELS = 10;

    std::vector<RGBDImage> images(10, RGBDImage(width, height));
    for (size_t image = 0; image < 10; image++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                images[image].setColor(x, y, 0, pixelSampler.getNext() / 100.0f);
                images[image].setColor(x, y, 1, pixelSampler.getNext() / 100.0f);
                images[image].setColor(x, y, 2, pixelSampler.getNext() / 100.0f);
                images[image].setDepth(x, y, Depth(depthSampler.getNext() / 10.0f));
            }
        }
        images[image].calculateIntegral();

        for (size_t sample = 0; sample < configuration.getSamplesPerImage(); sample++) {
            const LabelType label = image % NUM_LABELS;
            const uint16_t x = widthSampler.getNext();
            const uint16_t y = heightSampler.getNext();
            samples.push_back(PixelInstance(&images[image], label, x, y));
        }
    }

    ImageFeatureEvaluation featureEvaluation(0, configuration);
    RandomTree<PixelInstance, ImageFeatureFunction> node(0, 0, getPointers(samples), NUM_LABELS);

    ImageFeaturesAndThresholds<cuv::host_memory_space> features(configuration.getFeatureCount(),
            configuration.getThresholds(), boost::make_shared<cuv::default_allocator>());
    {
        utils::Profile profile("generateRandomFeatures (10 times)");

        std::vector<std::vector<const PixelInstance*> > batches = featureEvaluation.prepare(getPointers(samples), node,
                cuv::host_memory_space());

        for (int i = 0; i < 10; i++) {
            features = featureEvaluation.generateRandomFeatures(batches[0], i, true, cuv::host_memory_space());

            ImageFeaturesAndThresholds<cuv::host_memory_space> features2 =
                    featureEvaluation.generateRandomFeatures(batches[0], i, true, cuv::host_memory_space());

            assertEqual(features, features2, configuration);
        }
    }

    size_t hash = analyzeFeatures(configuration, features);

    BOOST_CHECK_EQUAL(4763250482941632261lu, hash);
}

BOOST_AUTO_TEST_CASE(testFeatureGenerationGPU) {

    clearImageCache();

    unsigned int samplesPerImage = 500;
    unsigned int featureCount = 2000;
    unsigned int minSampleCount = 32;
    int maxDepth = 15;
    uint16_t regionSize = 16;
    static const uint16_t THRESHOLDS = 50;
    int numThreads = NUM_THREADS;
    int maxImages = 10;
    int imageCacheSize = 10;
    unsigned int maxSamplesPerBatch = 5000;
    AccelerationMode accelerationMode = AccelerationMode::GPU_ONLY;

    TrainingConfiguration configuration(SEED, samplesPerImage, featureCount, minSampleCount, maxDepth, BOX_RADIUS,
            regionSize, THRESHOLDS, numThreads, maxImages, imageCacheSize, maxSamplesPerBatch, accelerationMode);

    static const int width = 400;
    static const int height = 400;

    std::vector<PixelInstance> samples;

    RandomSource randomSource(4711);
    Sampler widthSampler = randomSource.uniformSampler(width);
    Sampler heightSampler = randomSource.uniformSampler(height);

    Sampler pixelSampler = randomSource.uniformSampler(1000);
    Sampler depthSampler = randomSource.uniformSampler(10, 25);

    const int NUM_LABELS = 10;

    std::vector<RGBDImage> images(10, RGBDImage(width, height));
    for (size_t image = 0; image < images.size(); image++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                images[image].setColor(x, y, 0, pixelSampler.getNext() / 100.0f);
                images[image].setColor(x, y, 1, pixelSampler.getNext() / 100.0f);
                images[image].setColor(x, y, 2, pixelSampler.getNext() / 100.0f);
                images[image].setDepth(x, y, Depth(depthSampler.getNext() / 10.0f));
            }
        }
        images[image].calculateIntegral();

        for (size_t sample = 0; sample < configuration.getSamplesPerImage(); sample++) {
            const LabelType label = image % NUM_LABELS;
            const uint16_t x = widthSampler.getNext();
            const uint16_t y = heightSampler.getNext();
            samples.push_back(PixelInstance(&images[image], label, Depth(1.0), x, y));
        }
    }

    ImageFeatureEvaluation featureEvaluation(0, configuration);
    RandomTree<PixelInstance, ImageFeatureFunction> node(0, 0, getPointers(samples), NUM_LABELS);

    ImageFeaturesAndThresholds<cuv::host_memory_space> features(configuration.getFeatureCount(),
            configuration.getThresholds(), boost::make_shared<cuv::default_allocator>());
    {
        bool keepMutexLocked = false;
        std::vector<std::vector<const PixelInstance*> > batches = featureEvaluation.prepare(getPointers(samples),
                node, cuv::dev_memory_space(), keepMutexLocked);

        utils::Profile profile("generateRandomFeatures (10 times)");

        for (int i = 0; i < 10; i++) {

            ImageFeaturesAndThresholds<cuv::host_memory_space> features1(
                    featureEvaluation.generateRandomFeatures(batches[0], i, true, cuv::dev_memory_space()));

            ImageFeaturesAndThresholds<cuv::host_memory_space> features2(
                    featureEvaluation.generateRandomFeatures(batches[0], i, true, cuv::dev_memory_space()));

            assertEqual(features1, features2, configuration);

            features = features2;
        }
    }

    size_t hash = analyzeFeatures(configuration, features);

    BOOST_CHECK_EQUAL(4733702291017539369lu, hash);
}
BOOST_AUTO_TEST_SUITE_END()
