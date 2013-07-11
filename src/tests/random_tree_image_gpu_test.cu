#define BOOST_TEST_MODULE example

#include <assert.h>
#include <boost/functional/hash.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/test/included/unit_test.hpp>
#include <cuv/ndarray.hpp>

#include "random_tree_image_gpu.h"
#include "random_tree_image.h"
#include "score.h"
#include "test_common.h"
#include "utils.h"

using namespace curfil;

#define DUMP_IMAGE 0
static const int SEED = 4711;

class Fixture {
public:
    Fixture() {
        clearImageCache();
    }
};

BOOST_FIXTURE_TEST_SUITE(RandomTreeImageGPUTest, Fixture)

template<class W>
__global__
static void calculcateScoreKernel(ScoreType* result, const size_t numClasses,
        const W* leftClasses, const W* rightClasses, const unsigned int leftRightStride,
        const W* allClasses, const ScoreType totalLeft, const ScoreType totalRight) {
    ScoreType score = NormalizedInformationGainScore::calculateScore(numClasses, leftClasses, rightClasses,
            leftRightStride,
            allClasses, totalLeft, totalRight);
    *result = score;
}

static ScoreType scoreOnGPU(const size_t numClasses, const cuv::ndarray<int, cuv::host_memory_space>& leftClasses,
        const cuv::ndarray<int, cuv::host_memory_space>& rightClasses,
        const cuv::ndarray<int, cuv::host_memory_space>& allClasses,
        const ScoreType totalLeft, const ScoreType totalRight) {

    cuv::ndarray<ScoreType, cuv::dev_memory_space> result(1);

    const cuv::ndarray<int, cuv::dev_memory_space> leftClassesDevice(leftClasses);
    const cuv::ndarray<int, cuv::dev_memory_space> rightClassesDevice(rightClasses);
    const cuv::ndarray<int, cuv::dev_memory_space> allClassesDevice(allClasses);

    const unsigned int leftRightStride = leftClassesDevice.stride(0);
    BOOST_REQUIRE_EQUAL(leftRightStride, rightClassesDevice.stride(0));

    calculcateScoreKernel<<<1,1>>>(result.ptr(), numClasses, leftClassesDevice.ptr(), rightClassesDevice.ptr(),
            leftRightStride, allClassesDevice.ptr(), totalLeft, totalRight);

    cudaSafeCall(cudaThreadSynchronize());
    double res = result[0];
    return res;
}

static ScoreType scoreOnGPU(const size_t size, const WeightType* leftClass, const WeightType* rightClass,
        const WeightType* allClasses, const ScoreType totalLeft, const ScoreType totalRight) {

    cuv::ndarray<int, cuv::host_memory_space> leftClassArray(size);
    cuv::ndarray<int, cuv::host_memory_space> rightClassArray(size);
    cuv::ndarray<int, cuv::host_memory_space> allClassesArray(size);

    for (size_t i = 0; i < size; i++) {
        leftClassArray[i] = leftClass[i];
        rightClassArray[i] = rightClass[i];
        allClassesArray[i] = allClasses[i];
    }

    return scoreOnGPU(size, leftClassArray, rightClassArray, allClassesArray, totalLeft, totalRight);
}

BOOST_AUTO_TEST_CASE(testInformationGainScore) {

    const int numClasses = 2;
    cuv::ndarray<int, cuv::host_memory_space> left(numClasses);
    cuv::ndarray<int, cuv::host_memory_space> right(numClasses);
    cuv::ndarray<int, cuv::host_memory_space> allClass(numClasses);

    for (size_t num = 1; num < 10; num++) {

        // best case scenario: score=0
        left[0] = num;
        right[0] = num;

        left[1] = num;
        right[1] = num;

        allClass[0] = 2 * num;
        allClass[1] = 2 * num;

        ScoreType totalLeft = 2 * num;
        ScoreType totalRight = 2 * num;

        BOOST_REQUIRE_EQUAL(left.stride(0), 1);
        ScoreType score = NormalizedInformationGainScore::calculateScore(numClasses, left.ptr(), right.ptr(),
                left.stride(0), allClass.ptr(), totalLeft, totalRight);
        BOOST_CHECK_CLOSE(0, score, 0);
        BOOST_CHECK_CLOSE(score, scoreOnGPU(numClasses, left, right, allClass, totalLeft, totalRight), 1e-6);

        // best case scenario: score=1
        left[0] = 0;
        right[0] = 2 * num;

        left[1] = 2 * num;
        right[1] = 0;

        allClass[0] = 2 * num;
        allClass[1] = 2 * num;

        totalLeft = 2 * num;
        totalRight = 2 * num;

        BOOST_REQUIRE_EQUAL(left.stride(0), 1);
        score = NormalizedInformationGainScore::calculateScore(numClasses, left.ptr(), right.ptr(),
                left.stride(0), allClass.ptr(), totalLeft, totalRight);
        BOOST_CHECK_CLOSE(1, score, 0);
        BOOST_CHECK_CLOSE(score, scoreOnGPU(numClasses, left, right, allClass, totalLeft, totalRight), 1e-6);
    }

    left[0] = 5;
    right[0] = 3;

    left[1] = 8;
    right[1] = 1;

    allClass[0] = 8;
    allClass[1] = 9;

    double totalLeft = left[0] + left[1];
    double totalRight = right[0] + right[1];

    BOOST_REQUIRE_EQUAL(left.stride(0), 1);
    ScoreType score1 = NormalizedInformationGainScore::calculateScore(numClasses, left.ptr(), right.ptr(),
            left.stride(0), allClass.ptr(), totalLeft, totalRight);

    BOOST_CHECK_CLOSE(0.080185, score1, 1e-4);
    BOOST_CHECK_CLOSE(score1, scoreOnGPU(numClasses, left, right, allClass, totalLeft, totalRight), 1e-5);

    left[0] = 2;
    right[0] = 6;

    left[1] = 8;
    right[1] = 1;

    totalLeft = left[0] + left[1];
    totalRight = right[0] + right[1];

    BOOST_REQUIRE_EQUAL(left.stride(0), 1);
    ScoreType score2 = NormalizedInformationGainScore::calculateScore(numClasses, left.ptr(), right.ptr(),
            left.stride(0), allClass.ptr(), totalLeft, totalRight);

    BOOST_CHECK_GT(score2, score1);
    BOOST_CHECK_CLOSE(0.33339, score2, 1e-3);
    BOOST_CHECK_CLOSE(score2, scoreOnGPU(numClasses, left, right, allClass, totalLeft, totalRight), 1e-6);

    // case 1 (a real case)
    // histogram:       [ 86 241 291 3 267 ]
    // histogram left:  [ 56 241 290 3  18 ]
    // histogram right: [ 30   0   1 0 249 ]

    {
        const size_t size = 5;
        const WeightType all[] = { 86, 241, 291, 3, 267 };
        const WeightType left[] = { 56, 241, 290, 3, 18 };
        const WeightType right[] = { 30, 0, 1, 0, 249 };
        const unsigned int leftRightStride = 1;

        const size_t totalLeft = std::accumulate(left, left + size, 0);
        const size_t totalRight = std::accumulate(right, right + size, 0);

        BOOST_REQUIRE_EQUAL(totalLeft + totalRight, std::accumulate(all, all + size, 0));

        ScoreType score = NormalizedInformationGainScore::calculateScore(size, left, right,
                leftRightStride, all, totalLeft, totalRight);
        BOOST_CHECK_CLOSE(score, 0.491311, 1e-3);
        BOOST_CHECK_CLOSE(score, scoreOnGPU(size, left, right, all, totalLeft, totalRight), 1e-6);

        score = InformationGainScore::calculateScore(size, left, right,
                leftRightStride, all, totalLeft, totalRight);
        BOOST_CHECK_CLOSE(score, 0.690912, 1e-3);
    }

    {
        // case 2 (constructed, obviously)
        // histogram:       [ 50 100  50  0 100 ]
        // histogram left:  [ 50 100   0  0   0 ]
        // histogram right: [ 0    0  50  0 100 ]
        const size_t size = 5;
        const WeightType all[] = { 50, 100, 50, 0, 100 };
        const WeightType left[] = { 50, 100, 0, 0, 0 };
        const WeightType right[] = { 0, 0, 50, 0, 100 };
        const unsigned int leftRightStride = 1;

        const size_t totalLeft = std::accumulate(left, left + size, 0);
        const size_t totalRight = std::accumulate(right, right + size, 0);

        ScoreType score = NormalizedInformationGainScore::calculateScore(size, left, right, leftRightStride,
                all, totalLeft, totalRight);
        BOOST_CHECK_CLOSE(score, 0.68533, 1e-3);

        score = InformationGainScore::calculateScore(size, left, right, leftRightStride,
                all, totalLeft, totalRight);
        BOOST_CHECK_CLOSE(score, 1.0, 1e-3);
    }

}

template<class T>
static void updateHash(const boost::hash<size_t>& hasher, size_t& hash, const T& value) {
    // extract from boost headers
    hash ^= hasher(value) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
}

template<class T>
static size_t checkScores(const cuv::ndarray<ScoreType, cuv::host_memory_space>& scores, T numFeatures,
        T numThresholds) {
    BOOST_CHECK_EQUAL(2, static_cast<int>(scores.ndim()));
    BOOST_CHECK_EQUAL(static_cast<size_t>(numThresholds), static_cast<size_t>(scores.shape(0)));
    BOOST_CHECK_EQUAL(static_cast<size_t>(numFeatures), static_cast<size_t>(scores.shape(1)));

    size_t hash = 0;
    boost::hash<size_t> hasher;

    for (T feat = 0; feat < numFeatures; feat++) {
        for (T thresh = 0; thresh < numThresholds; thresh++) {
            const ScoreType score = scores(thresh, feat);

            BOOST_CHECK_GE(score, 0.0);
            BOOST_CHECK_LE(score, 1.0);

            updateHash(hasher, hash, score);
        }
    }

    return hash;
}

static size_t checkCounters(TrainingConfiguration& configuration,
        const cuv::ndarray<WeightType, cuv::dev_memory_space> countersDevice,
        const std::vector<PixelInstance>& samples) {

    const cuv::ndarray<WeightType, cuv::host_memory_space> counters(countersDevice);

    size_t hash = 0;
    boost::hash<size_t> hasher;

    std::map<size_t, size_t> samplesPerLabel;
    for (size_t sample = 0; sample < samples.size(); sample++) {
        samplesPerLabel[samples[sample].getLabel()]++;
    }

    size_t numLabels = samplesPerLabel.size();

    assert(numLabels > 0);

    const size_t features = configuration.getFeatureCount();
    const size_t thresholds = configuration.getThresholds();

    BOOST_CHECK_EQUAL(4, static_cast<int>(counters.ndim()));
    BOOST_CHECK_EQUAL(features, static_cast<size_t>(counters.shape(0)));
    BOOST_CHECK_EQUAL(thresholds, static_cast<size_t>(counters.shape(1)));
    BOOST_CHECK_EQUAL(numLabels, static_cast<size_t>(counters.shape(2)));
    BOOST_CHECK_EQUAL(2lu, static_cast<size_t>(counters.shape()[3]));

    for (size_t label = 0; label < numLabels; label++) {
        for (size_t thresh = 0; thresh < thresholds; thresh++) {
            for (size_t feat = 0; feat < features; feat++) {

                const size_t left = counters(feat, thresh, label, 0);
                const size_t right = counters(feat, thresh, label, 1);

                const size_t numSamples = samplesPerLabel[label];
                BOOST_CHECK_EQUAL(numSamples, left + right);

                updateHash(hasher, hash, left);
                updateHash(hasher, hash, right);
            }
        }
    }

    return hash;
}

BOOST_AUTO_TEST_CASE(testDepthFeatureSimple) {

    const int NUM_FEAT = 1;
    const int NUM_THRESH = 100;

    unsigned int samplesPerImage = 500;
    unsigned int minSampleCount = 32;
    int maxDepth = 15;
    uint16_t boxRadius = 127;
    uint16_t regionSize = 16;
    static const int NUM_THREADS = 1;
    static const int maxImages = 0;
    static const int imageCacheSize = 1;
    unsigned int maxSamplesPerBatch = 5000;
    AccelerationMode accelerationMode = GPU_AND_CPU_COMPARE;

    TrainingConfiguration configuration(SEED, samplesPerImage, NUM_FEAT, minSampleCount, maxDepth, boxRadius,
            regionSize, NUM_THRESH, NUM_THREADS, maxImages, imageCacheSize, maxSamplesPerBatch, accelerationMode);

    ImageFeatureEvaluation featureFunction(0, configuration);

    static const int width = 16;
    static const int height = 20;

    std::vector<RGBDImage> images(1, RGBDImage(width, height));
    std::vector<PixelInstance> samples;

    ImageFeaturesAndThresholds<cuv::dev_memory_space> featuresAndThresholds(NUM_FEAT, NUM_THRESH,
            boost::make_shared<cuv::default_allocator>());

    for (int i = 0; i < NUM_THRESH; i++) {
        featuresAndThresholds.thresholds()(i, 0) = (i - 50) / 10.0f;
    }

    featuresAndThresholds.types()[0] = DEPTH;

    featuresAndThresholds.offset1X()[0] = 1;
    featuresAndThresholds.offset1Y()[0] = 1;

    featuresAndThresholds.region1X()[0] = 2;
    featuresAndThresholds.region1Y()[0] = 2;

    featuresAndThresholds.offset2X()[0] = -3;
    featuresAndThresholds.offset2Y()[0] = -1;

    featuresAndThresholds.region2X()[0] = 1;
    featuresAndThresholds.region2Y()[0] = 1;

    featuresAndThresholds.channel1()[0] = 0;
    featuresAndThresholds.channel2()[0] = 0;

    const int NUM_LABELS = 2;

    samples.push_back(PixelInstance(&images[0], 0, Depth(1.0), 6, 3));
    samples.push_back(PixelInstance(&images[0], 1, Depth(1.0), 6, 3));

    RandomTree<PixelInstance, ImageFeatureFunction> node(0, 0, getPointers(samples), NUM_LABELS);
    cuv::ndarray<WeightType, cuv::dev_memory_space> histogram(node.getHistogram());

    {
        std::vector<std::vector<const PixelInstance*> > batches = featureFunction.prepare(getPointers(samples), node,
                cuv::dev_memory_space(), false);

        cuv::ndarray<FeatureResponseType, cuv::host_memory_space> featureResponses;

        cuv::ndarray<WeightType, cuv::dev_memory_space> counters =
                featureFunction.calculateFeatureResponsesAndHistograms(node, batches, featuresAndThresholds,
                        &featureResponses);

        BOOST_CHECK(isnan(static_cast<FeatureResponseType>(featureResponses(0, 0))));

        cuv::ndarray<ScoreType, cuv::host_memory_space> scores(featureFunction.calculateScores(counters,
                featuresAndThresholds, histogram));

        checkScores(scores, NUM_FEAT, NUM_THRESH);
    }

    images[0].reset();
    clearImageCache();

    images[0].setDepth(7, 3, Depth(1.5f));
    images[0].setDepth(7, 4, Depth(1.7f));
    images[0].setDepth(8, 4, Depth(4.5f));
    images[0].setDepth(3, 2, Depth(3.9f));

#if DUMP_IMAGE
    image.dumpDepth(std::cout);
#endif

    images[0].calculateIntegral();

#if DUMP_IMAGE
    CURFIL_INFO("integral:");
    image.dumpDepth(std::cout);
#endif

    {
        std::vector<std::vector<const PixelInstance*> > batches = featureFunction.prepare(getPointers(samples),
                node, cuv::dev_memory_space(), false);

        cuv::ndarray<FeatureResponseType, cuv::host_memory_space> featureResponses;

        cuv::ndarray<WeightType, cuv::dev_memory_space> counters =
                featureFunction.calculateFeatureResponsesAndHistograms(node, batches, featuresAndThresholds,
                        &featureResponses);

        cuv::ndarray<ScoreType, cuv::host_memory_space> scores(featureFunction.calculateScores(counters,
                featuresAndThresholds, histogram));

        checkCounters(configuration, counters, samples);

        checkScores(scores, NUM_FEAT, NUM_THRESH);

        BOOST_CHECK_CLOSE((1.5 + 1.7 + 4.5) / 3 - 3.9, static_cast<FeatureResponseType>(featureResponses(0, 0)), 1e-6);

        for (int label = 0; label < NUM_LABELS; label++) {
            for (int thresh = 0; thresh < NUM_THRESH; thresh++) {
                for (int feat = 0; feat < NUM_FEAT; feat++) {
                    BOOST_CHECK_EQUAL(1,
                            static_cast<int>(counters(feat, thresh, label, 0) + counters(feat, thresh, label, 1)));
                    BOOST_CHECK_EQUAL(static_cast<unsigned int>(thresh >= 37),
                            static_cast<unsigned int>(counters(feat, thresh, label, 0)));
                }
            }
        }

        images[0].reset();
        clearImageCache();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                images[0].setDepth(x, y, Depth(1.0f));
            }
        }
        images[0].calculateIntegral();
#if DUMP_IMAGE
        image.dumpDepth(std::cout);
#endif

        {
            std::vector<std::vector<const PixelInstance*> > batches = featureFunction.prepare(getPointers(samples),
                    node, cuv::dev_memory_space(), false);

            cuv::ndarray<FeatureResponseType, cuv::host_memory_space> featureResponses;

            cuv::ndarray<WeightType, cuv::dev_memory_space> counters =
                    featureFunction.calculateFeatureResponsesAndHistograms(node, batches, featuresAndThresholds,
                            &featureResponses);

            cuv::ndarray<ScoreType, cuv::host_memory_space> scores(featureFunction.calculateScores(counters,
                    featuresAndThresholds, histogram));

            BOOST_CHECK_CLOSE(0, static_cast<FeatureResponseType>(featureResponses(0, 0)), 0);

            checkScores(scores, NUM_FEAT, NUM_THRESH);

            checkCounters(configuration, counters, samples);

            for (int label = 0; label < NUM_LABELS; label++) {
                for (int thresh = 0; thresh < 100; thresh++) {
                    for (int feat = 0; feat < NUM_FEAT; feat++) {
                        BOOST_CHECK_EQUAL(1,
                                static_cast<int>(counters(feat, thresh, label, 0) + counters(feat, thresh, label, 1)));
                        BOOST_CHECK_EQUAL(static_cast<unsigned int>(thresh >= 50),
                                static_cast<unsigned int>(counters(feat, thresh, label, 0)));
                    }
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testColorFeatureSimple) {

    unsigned int samplesPerImage = 500;
    unsigned int NUM_FEAT = 1;
    unsigned int minSampleCount = 32;
    int maxDepth = 15;
    uint16_t boxRadius = 127;
    uint16_t regionSize = 16;
    static const uint16_t NUM_THRESH = 3;
    static const int NUM_THREADS = 1;
    static const int maxImages = 0;
    static const int imageCacheSize = 1;
    unsigned int maxSamplesPerBatch = 5000;
    AccelerationMode accelerationMode = GPU_AND_CPU_COMPARE;

    TrainingConfiguration configuration(SEED, samplesPerImage, NUM_FEAT, minSampleCount, maxDepth, boxRadius,
            regionSize, NUM_THRESH, NUM_THREADS, maxImages, imageCacheSize, maxSamplesPerBatch, accelerationMode);

    ImageFeatureEvaluation featureFunction(0, configuration);

    static const int width = 16;
    static const int height = 20;

    std::vector<RGBDImage> images(1, RGBDImage(width, height));
    std::vector<PixelInstance> samples;

    ImageFeaturesAndThresholds<cuv::dev_memory_space> featuresAndThresholds(NUM_FEAT, NUM_THRESH,
            boost::make_shared<cuv::default_allocator>());

    featuresAndThresholds.types()[0] = COLOR;

    featuresAndThresholds.offset1X()[0] = 3;
    featuresAndThresholds.offset1Y()[0] = -2;

    featuresAndThresholds.region1X()[0] = 4;
    featuresAndThresholds.region1Y()[0] = 2;

    featuresAndThresholds.offset2X()[0] = -3;
    featuresAndThresholds.offset2Y()[0] = -2;

    featuresAndThresholds.region2X()[0] = 1;
    featuresAndThresholds.region2Y()[0] = 2;

    featuresAndThresholds.channel1()[0] = 0;
    featuresAndThresholds.channel2()[0] = 0;

    featuresAndThresholds.thresholds()(0, 0) = -1.0f;
    featuresAndThresholds.thresholds()(1, 0) = 0.0f;
    featuresAndThresholds.thresholds()(2, 0) = 1.0f;

    const int NUM_LABELS = 2;

    samples.push_back(PixelInstance(&images[0], 0, Depth(1.0), 6, 4));
    samples.push_back(PixelInstance(&images[0], 1, Depth(1.0), 6, 4));

    RandomTree<PixelInstance, ImageFeatureFunction> node(0, 0, getPointers(samples), NUM_LABELS);
    cuv::ndarray<WeightType, cuv::dev_memory_space> histogram(node.getHistogram());

    {
        std::vector<std::vector<const PixelInstance*> > batches = featureFunction.prepare(getPointers(samples), node,
                cuv::dev_memory_space(), false);

        cuv::ndarray<FeatureResponseType, cuv::host_memory_space> featureResponses;

        cuv::ndarray<WeightType, cuv::dev_memory_space> counters =
                featureFunction.calculateFeatureResponsesAndHistograms(node, batches, featuresAndThresholds,
                        &featureResponses);

        BOOST_CHECK_CLOSE(0, static_cast<FeatureResponseType>(featureResponses(0, 0)), 0);

        checkCounters(configuration, counters, samples);

        assert(static_cast<int>(samples.size()) == NUM_LABELS);
        for (size_t label = 0; label < samples.size(); label++) {
            BOOST_CHECK_EQUAL(0, static_cast<int>(counters(0, 0, label, 0)));
            BOOST_CHECK_EQUAL(1, static_cast<int>(counters(0, 0, label, 1)));

            BOOST_CHECK_EQUAL(1, static_cast<int>(counters(0, 1, label, 0)));
            BOOST_CHECK_EQUAL(0, static_cast<int>(counters(0, 1, label, 1)));

            BOOST_CHECK_EQUAL(1, static_cast<int>(counters(0, 2, label, 0)));
            BOOST_CHECK_EQUAL(0, static_cast<int>(counters(0, 2, label, 1)));
        }
    }

    images[0].reset();
    clearImageCache();

    images[0].setColor(7, 2, 0, 0.5f);

#if DUMP_IMAGE
    image.dump(std::cout);
#endif

    images[0].calculateIntegral();

#if DUMP_IMAGE
    std::cout << "integral" << std::endl;
    image.dump(std::cout);
#endif

    {
        std::vector<std::vector<const PixelInstance*> > batches = featureFunction.prepare(getPointers(samples),
                node, cuv::dev_memory_space(), false);

        cuv::ndarray<FeatureResponseType, cuv::host_memory_space> featureResponses;

        cuv::ndarray<WeightType, cuv::dev_memory_space> counters =
                featureFunction.calculateFeatureResponsesAndHistograms(node, batches, featuresAndThresholds,
                        &featureResponses);

        checkCounters(configuration, counters, samples);

        BOOST_CHECK_CLOSE(0.5, static_cast<FeatureResponseType>(featureResponses(0, 0)), 0);

        for (size_t label = 0; label < samples.size(); label++) {
            BOOST_CHECK_EQUAL(0, static_cast<int>(counters(0, 0, label, 0)));
            BOOST_CHECK_EQUAL(1, static_cast<int>(counters(0, 0, label, 1)));
            BOOST_CHECK_EQUAL(0, static_cast<int>(counters(0, 1, label, 0)));
            BOOST_CHECK_EQUAL(1, static_cast<int>(counters(0, 1, label, 1)));
            BOOST_CHECK_EQUAL(1, static_cast<int>(counters(0, 2, label, 0)));
            BOOST_CHECK_EQUAL(0, static_cast<int>(counters(0, 2, label, 1)));
        }
    }

    images[0].reset();
    clearImageCache();

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            images[0].setColor(x, y, 0, 1.0f);
        }
    }
    images[0].calculateIntegral();
#if DUMP_IMAGE
    image.dump(std::cout);
#endif

    {
        std::vector<std::vector<const PixelInstance*> > batches = featureFunction.prepare(getPointers(samples),
                node, cuv::dev_memory_space(), false);

        cuv::ndarray<FeatureResponseType, cuv::host_memory_space> featureResponses;

        cuv::ndarray<WeightType, cuv::dev_memory_space> counters =
                featureFunction.calculateFeatureResponsesAndHistograms(node, batches, featuresAndThresholds,
                        &featureResponses);

        BOOST_CHECK_CLOSE(8 * 4 - 4 * 2.0, static_cast<FeatureResponseType>(featureResponses(0, 0)), 0);

        checkCounters(configuration, counters, samples);

        for (size_t label = 0; label < samples.size(); label++) {
            BOOST_CHECK_EQUAL(0, static_cast<int>(counters(0, 0, label, 0)));
            BOOST_CHECK_EQUAL(1, static_cast<int>(counters(0, 0, label, 1)));
            BOOST_CHECK_EQUAL(0, static_cast<int>(counters(0, 1, label, 0)));
            BOOST_CHECK_EQUAL(1, static_cast<int>(counters(0, 1, label, 1)));
            BOOST_CHECK_EQUAL(0, static_cast<int>(counters(0, 2, label, 0)));
            BOOST_CHECK_EQUAL(1, static_cast<int>(counters(0, 2, label, 1)));
        }
    }
}

BOOST_AUTO_TEST_CASE(testColorFeatureComplex) {

    const size_t NUM_THRESH = 2;
    const size_t NUM_FEAT = 3;
    const int maxImages = 5;
    const int imageCacheSize = 5; // make sure the cache is at least as big as #images

    unsigned int samplesPerImage = 500;
    unsigned int minSampleCount = 32;
    int maxDepth = 15;
    uint16_t boxRadius = 127;
    uint16_t regionSize = 16;
    static const int NUM_THREADS = 1;
    unsigned int maxSamplesPerBatch = 5000;
    AccelerationMode accelerationMode = GPU_AND_CPU_COMPARE;

    TrainingConfiguration configuration(SEED, samplesPerImage, NUM_FEAT, minSampleCount, maxDepth, boxRadius,
            regionSize, NUM_THRESH, NUM_THREADS, maxImages, imageCacheSize, maxSamplesPerBatch, accelerationMode);

    ImageFeatureEvaluation featureFunction(0, configuration);

    const int width = 12;
    const int height = 15;

    std::vector<RGBDImage> images(2, RGBDImage(width, height));
    std::vector<PixelInstance> samples;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                float v = 100 * c + y * width + x;
                images[0].setColor(x, y, c, v);
                images[1].setColor(x, y, c, v / 2.0f);
            }
        }
    }

    BOOST_CHECK_CLOSE(179, images[0].getColor(width - 1, height - 1, 0), 0);
    BOOST_CHECK_CLOSE(279, images[0].getColor(width - 1, height - 1, 1), 0);
    BOOST_CHECK_CLOSE(379, images[0].getColor(width - 1, height - 1, 2), 0);

    BOOST_CHECK_CLOSE(179 / 2.0, images[1].getColor(width - 1, height - 1, 0), 0);
    BOOST_CHECK_CLOSE(279 / 2.0, images[1].getColor(width - 1, height - 1, 1), 0);
    BOOST_CHECK_CLOSE(379 / 2.0, images[1].getColor(width - 1, height - 1, 2), 0);

    images[0].calculateIntegral();
    images[1].calculateIntegral();

    BOOST_CHECK_CLOSE(16110, images[0].getColor(width - 1, height - 1, 0), 0);
    BOOST_CHECK_CLOSE(34110, images[0].getColor(width - 1, height - 1, 1), 0);
    BOOST_CHECK_CLOSE(52110, images[0].getColor(width - 1, height - 1, 2), 0);

    BOOST_CHECK_CLOSE(16110 / 2.0, images[1].getColor(width - 1, height - 1, 0), 0);
    BOOST_CHECK_CLOSE(34110 / 2.0, images[1].getColor(width - 1, height - 1, 1), 0);
    BOOST_CHECK_CLOSE(52110 / 2.0, images[1].getColor(width - 1, height - 1, 2), 0);

    ImageFeaturesAndThresholds<cuv::dev_memory_space> featuresAndThresholds(NUM_FEAT, NUM_THRESH,
            boost::make_shared<cuv::default_allocator>());

    featuresAndThresholds.types()[0] = COLOR;
    featuresAndThresholds.offset1X()[0] = 2;
    featuresAndThresholds.offset1Y()[0] = -1;
    featuresAndThresholds.region1X()[0] = 2;
    featuresAndThresholds.region1Y()[0] = 1;
    featuresAndThresholds.offset2X()[0] = -3;
    featuresAndThresholds.offset2Y()[0] = 4;
    featuresAndThresholds.region2X()[0] = 1;
    featuresAndThresholds.region2Y()[0] = 2;
    featuresAndThresholds.channel1()[0] = 0;
    featuresAndThresholds.channel2()[0] = 2;

    featuresAndThresholds.types()[1] = COLOR;
    featuresAndThresholds.offset1X()[1] = 2;
    featuresAndThresholds.offset1Y()[1] = -1;
    featuresAndThresholds.region1X()[1] = 2;
    featuresAndThresholds.region1Y()[1] = 2;
    featuresAndThresholds.offset2X()[1] = -3;
    featuresAndThresholds.offset2Y()[1] = 4;
    featuresAndThresholds.region2X()[1] = 1;
    featuresAndThresholds.region2Y()[1] = 1;
    featuresAndThresholds.channel1()[1] = 1;
    featuresAndThresholds.channel2()[1] = 2;

    featuresAndThresholds.types()[2] = COLOR;
    featuresAndThresholds.offset1X()[2] = -2;
    featuresAndThresholds.offset1Y()[2] = 1;
    featuresAndThresholds.region1X()[2] = 3;
    featuresAndThresholds.region1Y()[2] = 1;
    featuresAndThresholds.offset2X()[2] = 3;
    featuresAndThresholds.offset2Y()[2] = -4;
    featuresAndThresholds.region2X()[2] = 3;
    featuresAndThresholds.region2Y()[2] = 3;
    featuresAndThresholds.channel1()[2] = 1;
    featuresAndThresholds.channel2()[2] = 0;

    featuresAndThresholds.thresholds()(0, 0) = 0.0f;
    featuresAndThresholds.thresholds()(1, 0) = -500.0f;

    featuresAndThresholds.thresholds()(0, 1) = -300.0f;
    featuresAndThresholds.thresholds()(1, 1) = 0.0f;

    featuresAndThresholds.thresholds()(0, 2) = 0.0f;
    featuresAndThresholds.thresholds()(1, 2) = 500.0f;

    const int NUM_LABELS = 2;

    samples.push_back(PixelInstance(&images[0], 0, Depth(1.0), 6, 4));
    samples.push_back(PixelInstance(&images[1], 0, Depth(2.0), 6, 4));
    samples.push_back(PixelInstance(&images[0], 1, Depth(1.5), 5, 5));
    samples.push_back(PixelInstance(&images[1], 1, Depth(3.1), 3, 4));

    RandomTree<PixelInstance, ImageFeatureFunction> node(0, 0, getPointers(samples), NUM_LABELS);
    cuv::ndarray<WeightType, cuv::dev_memory_space> histogram(node.getHistogram());

    // 2 images, 3 features, 4 samples

    std::vector<std::vector<const PixelInstance*> > batches = featureFunction.prepare(getPointers(samples),
            node, cuv::dev_memory_space(), false);

    cuv::ndarray<FeatureResponseType, cuv::host_memory_space> featureResponses;

    cuv::ndarray<WeightType, cuv::dev_memory_space> counters =
            featureFunction.calculateFeatureResponsesAndHistograms(node, batches, featuresAndThresholds,
                    &featureResponses);

    cuv::ndarray<ScoreType, cuv::host_memory_space> scores(
            featureFunction.calculateScores(counters,
                    featuresAndThresholds, histogram));

    BOOST_CHECK_EQUAL(2, static_cast<int>(featureResponses.ndim()));
    BOOST_CHECK_EQUAL(3, static_cast<int>(featureResponses.shape(0)));
    BOOST_CHECK_EQUAL(4, static_cast<int>(featureResponses.shape(1)));

    checkScores(scores, NUM_FEAT, NUM_THRESH);

    // values verified by manual calculation
    // sample 0, feat 0
    BOOST_CHECK_CLOSE(-2040, static_cast<FeatureResponseType>(featureResponses(0, 0)), 0);
    // sample 0, feat 1
    BOOST_CHECK_CLOSE(1186, static_cast<FeatureResponseType>(featureResponses(1, 0)), 0);
    // sample 0, feat 2
    BOOST_CHECK(isnan(static_cast<FeatureResponseType>(featureResponses(2, 0))));
    // sample 1, feat 0
    BOOST_CHECK_CLOSE(-444, static_cast<FeatureResponseType>(featureResponses(0, 1)), 0);
    // sample 1, feat 1
    BOOST_CHECK_CLOSE(-244, static_cast<FeatureResponseType>(featureResponses(1, 1)), 0);
    // sample 1, feat 2
    BOOST_CHECK_CLOSE(244, static_cast<FeatureResponseType>(featureResponses(2, 1)), 0);
    // sample 2, feat 0
    BOOST_CHECK_CLOSE(-884, static_cast<FeatureResponseType>(featureResponses(0, 2)), 0);
    // sample 2, feat 1
    BOOST_CHECK_CLOSE(-484, static_cast<FeatureResponseType>(featureResponses(1, 2)), 0);
    // sample 2, feat 2
    BOOST_CHECK_CLOSE(572, static_cast<FeatureResponseType>(featureResponses(2, 2)), 0);
    // sample 3, feat 0
    BOOST_CHECK_CLOSE(-424, static_cast<FeatureResponseType>(featureResponses(0, 3)), 0);
    // sample 3, feat 1
    BOOST_CHECK_CLOSE(-224, static_cast<FeatureResponseType>(featureResponses(1, 3)), 0);
    // sample 3, feat 2
    BOOST_CHECK_CLOSE(224, static_cast<FeatureResponseType>(featureResponses(2, 3)), 0);

    checkCounters(configuration, counters, samples);

    // -2040    sample 0, feat 0  →  0
    //  -444    sample 1, feat 0  →  0
    //  -884    sample 2, feat 0  →  1
    //  -424    sample 3, feat 0  →  1

    //  1186    sample 0, feat 1  →  0
    //  -244    sample 1, feat 1  →  0
    //  -484    sample 2, feat 1  →  1
    //  -224    sample 3, feat 1  →  1

    //  1551    sample 0, feat 2  →  0
    //   244    sample 1, feat 2  →  0
    //   572    sample 2, feat 2  →  1
    //   224    sample 3, feat 2  →  1

    // feat 0, thresh 0 (0.0f)
    BOOST_CHECK_EQUAL(2, static_cast<int>(counters(0, 0, 0, 0)));
    BOOST_CHECK_EQUAL(0, static_cast<int>(counters(0, 0, 0, 1)));
    BOOST_CHECK_EQUAL(2, static_cast<int>(counters(0, 0, 1, 0)));
    BOOST_CHECK_EQUAL(0, static_cast<int>(counters(0, 0, 1, 1)));

    // feat 0, thresh 1 (-500.0f)
    BOOST_CHECK_EQUAL(1, static_cast<int>(counters(0, 1, 0, 0)));
    BOOST_CHECK_EQUAL(1, static_cast<int>(counters(0, 1, 0, 1)));
    BOOST_CHECK_EQUAL(1, static_cast<int>(counters(0, 1, 1, 0)));
    BOOST_CHECK_EQUAL(1, static_cast<int>(counters(0, 1, 1, 1)));

    // feat 1, thresh 0 (-300.0f)
    BOOST_CHECK_EQUAL(0, static_cast<int>(counters(1, 0, 0, 0)));
    BOOST_CHECK_EQUAL(2, static_cast<int>(counters(1, 0, 0, 1)));
    BOOST_CHECK_EQUAL(1, static_cast<int>(counters(1, 0, 1, 0)));
    BOOST_CHECK_EQUAL(1, static_cast<int>(counters(1, 0, 1, 1)));

    // feat 1, thresh 1 (0.0f)
    BOOST_CHECK_EQUAL(1, static_cast<int>(counters(1, 1, 0, 0)));
    BOOST_CHECK_EQUAL(1, static_cast<int>(counters(1, 1, 0, 1)));
    BOOST_CHECK_EQUAL(2, static_cast<int>(counters(1, 1, 1, 0)));
    BOOST_CHECK_EQUAL(0, static_cast<int>(counters(1, 1, 1, 1)));

    // feat 2, thresh 0 (0.0f)
    BOOST_CHECK_EQUAL(0, static_cast<int>(counters(2, 0, 0, 0)));
    BOOST_CHECK_EQUAL(2, static_cast<int>(counters(2, 0, 0, 1)));
    BOOST_CHECK_EQUAL(0, static_cast<int>(counters(2, 0, 1, 0)));
    BOOST_CHECK_EQUAL(2, static_cast<int>(counters(2, 0, 1, 1)));

    // feat 2, thresh 1 (+500.0f)
    BOOST_CHECK_EQUAL(1, static_cast<int>(counters(2, 1, 0, 0)));
    BOOST_CHECK_EQUAL(1, static_cast<int>(counters(2, 1, 0, 1)));
    BOOST_CHECK_EQUAL(1, static_cast<int>(counters(2, 1, 1, 0)));
    BOOST_CHECK_EQUAL(1, static_cast<int>(counters(2, 1, 1, 1)));
}

BOOST_AUTO_TEST_CASE(testColorFeatureManySamples) {

    const int NUM_FEAT = 1000;
    const int NUM_THRESH = 50;
    unsigned int samplesPerImage = 100;

    unsigned int minSampleCount = 32;
    int maxDepth = 15;
    uint16_t boxRadius = 127;
    uint16_t regionSize = 50;
    static const int NUM_THREADS = 1;
    static const int maxImages = 5;
    static const int imageCacheSize = 5;
    unsigned int maxSamplesPerBatch = 100000;
    AccelerationMode accelerationMode = GPU_AND_CPU_COMPARE;

    TrainingConfiguration configuration(SEED, samplesPerImage, NUM_FEAT, minSampleCount, maxDepth, boxRadius,
            regionSize, NUM_THRESH, NUM_THREADS, maxImages, imageCacheSize, maxSamplesPerBatch, accelerationMode);

    ImageFeatureEvaluation featureFunction(0, configuration);

    std::vector<PixelInstance> samples;

    const int width = 640;
    const int height = 480;

    std::vector<RGBDImage> images(10, RGBDImage(width, height));
    for (size_t image = 0; image < images.size(); image++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < 3; c++) {
                    float v = 10000 * image + 100 * c + y * width + x;
                    images[image].setColor(x, y, c, v);
                }
            }
        }

        images[image].calculateIntegral();
    }

    const size_t NUM_LABELS = 10;

    const int NUM_SAMPLES = samplesPerImage * images.size();
    for (int i = 0; i < NUM_SAMPLES; i++) {
        PixelInstance sample(
                &images.at(i / (NUM_SAMPLES / images.size())),   // image
                i / 100,   // label
                Depth((i % 20) / 10.0 + 1.0),   // depth
                i % width,   // x
                i % height   // y
                        );

        samples.push_back(sample);
    }

    RandomTree<PixelInstance, ImageFeatureFunction> node(0, 0, getPointers(samples), NUM_LABELS);
    cuv::ndarray<WeightType, cuv::dev_memory_space> histogram(node.getHistogram());

    {
        std::vector<std::vector<const PixelInstance*> > batches = featureFunction.prepare(getPointers(samples),
                node, cuv::dev_memory_space(), false);

        ImageFeaturesAndThresholds<cuv::dev_memory_space> featuresAndThresholds =
                featureFunction.generateRandomFeatures(batches[0], configuration.getRandomSeed(),
                        true, cuv::dev_memory_space());

        cuv::ndarray<WeightType, cuv::dev_memory_space> counters =
                featureFunction.calculateFeatureResponsesAndHistograms(
                        node, batches, featuresAndThresholds);

        cuv::ndarray<ScoreType, cuv::host_memory_space> scores(
                featureFunction.calculateScores(counters,
                        featuresAndThresholds, histogram));

        size_t scoreHash = checkScores(scores, NUM_FEAT, NUM_THRESH);
        size_t counterHash = checkCounters(configuration, counters, samples);

        // magic number. used to check for regressions
        BOOST_CHECK_EQUAL(4437303196209240250lu, counterHash);
        BOOST_CHECK_EQUAL(13702092111133522162lu, scoreHash);
    }
}

static void checkNode(boost::shared_ptr<const RandomTree<PixelInstance, ImageFeatureFunction> > node,
        const boost::shared_ptr<const TreeNodes>& treeData,
        const SplitFunction<PixelInstance, ImageFeatureFunction>* split = 0) {

    const size_t numLabels = node->getHistogram().size();
    const size_t nodeNr = node->getNodeId();
    assert(nodeNr - node->getTreeId() < treeData->numNodes());

    TreeNodeData data = getTreeNode(nodeNr, treeData);

    if (node->isLeaf()) {
        BOOST_CHECK_EQUAL(-1, static_cast<int>(data.leftNodeOffset));
        BOOST_CHECK(isnan(static_cast<float>(data.threshold)));
        for (size_t label = 0; label < numLabels; label++) {
            BOOST_CHECK_EQUAL(static_cast<float>(node->getNormalizedHistogram()[label]),
                    static_cast<float>(data.histogram(label)));
        }
    } else {
        BOOST_REQUIRE(split);
        const ImageFeatureFunction& feature = split->getFeature();
        const float expectedThreshold = split->getThreshold();
        const int expectedLeftNodeOffset = node->getLeft()->getNodeId() - node->getNodeId();

        BOOST_CHECK_EQUAL(expectedLeftNodeOffset, static_cast<int>(data.leftNodeOffset));
        BOOST_CHECK_EQUAL(expectedThreshold, static_cast<float>(data.threshold));
        BOOST_CHECK_EQUAL(static_cast<int>(feature.getType()), static_cast<int>(data.type));
        BOOST_CHECK_EQUAL(feature.getOffset1().getX(), static_cast<int>(data.offset1X));
        BOOST_CHECK_EQUAL(feature.getOffset1().getY(), static_cast<int>(data.offset1Y));
        BOOST_CHECK_EQUAL(feature.getRegion1().getX(), static_cast<int>(data.region1X));
        BOOST_CHECK_EQUAL(feature.getRegion1().getY(), static_cast<int>(data.region1Y));
        BOOST_CHECK_EQUAL(feature.getOffset2().getX(), static_cast<int>(data.offset2X));
        BOOST_CHECK_EQUAL(feature.getOffset2().getY(), static_cast<int>(data.offset2Y));
        BOOST_CHECK_EQUAL(feature.getRegion2().getX(), static_cast<int>(data.region2X));
        BOOST_CHECK_EQUAL(feature.getRegion2().getY(), static_cast<int>(data.region2Y));
        BOOST_CHECK_EQUAL(feature.getChannel1(), static_cast<uint8_t>(data.channel1));
        BOOST_CHECK_EQUAL(feature.getChannel2(), static_cast<uint8_t>(data.channel2));
    }
}

BOOST_AUTO_TEST_CASE(testRecallOnGPU) {

    const int NUM_FEAT = 1000;
    const int NUM_THRESH = 50;
    unsigned int samplesPerImage = 100;

    unsigned int minSampleCount = 32;
    int maxDepth = 15;
    uint16_t boxRadius = 127;
    uint16_t regionSize = 50;
    static const int NUM_THREADS = 1;
    static const int maxImages = 5;
    static const int imageCacheSize = 5;
    unsigned int maxSamplesPerBatch = 100000;
    AccelerationMode accelerationMode = GPU_AND_CPU_COMPARE;

    TrainingConfiguration configuration(SEED, samplesPerImage, NUM_FEAT, minSampleCount, maxDepth, boxRadius,
            regionSize, NUM_THRESH, NUM_THREADS, maxImages, imageCacheSize, maxSamplesPerBatch, accelerationMode);

    ImageFeatureEvaluation featureFunction(0, configuration);

    std::vector<PixelInstance> samples;

    const int width = 640;
    const int height = 480;

    std::vector<RGBDImage> images(10, RGBDImage(width, height));
    for (size_t image = 0; image < images.size(); image++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < 3; c++) {
                    float v = 10000 * image + 100 * c + y * width + x;
                    images[image].setColor(x, y, c, v);
                }
            }
        }

        images[image].calculateIntegral();
    }

    const size_t NUM_LABELS = 3;

    const int NUM_SAMPLES = samplesPerImage * images.size();
    for (int i = 0; i < NUM_SAMPLES; i++) {
        PixelInstance sample(
                &images.at(i / (NUM_SAMPLES / images.size())), // image
                (i / 100) % NUM_LABELS, // label
                Depth((i % 20) / 10.0 + 0.1), // depth
                i % width, // x
                i % height // y
                        );

        samples.push_back(sample);
    }

    /**
     *          n0
     *         /  \
     *        /    \
     *       n1    n2
     *      / \
     *     /   \
     *    n3   n4
     */
    boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> > n0 =
            boost::make_shared<RandomTree<PixelInstance, ImageFeatureFunction> >(0, 0, getPointers(samples),
                    NUM_LABELS);

    std::vector<WeightType> histN1(NUM_LABELS, 0);
    histN1[0] = 10;
    histN1[1] = 10;
    histN1[2] = 10;
    boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> > n1 =
            boost::make_shared<RandomTree<PixelInstance, ImageFeatureFunction> >(1, 1, n0, histN1);

    std::vector<WeightType> histN2(NUM_LABELS, 0);
    histN2[0] = 60;
    histN2[1] = 60;
    histN2[2] = 20;
    boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> > n2 =
            boost::make_shared<RandomTree<PixelInstance, ImageFeatureFunction> >(2, 1, n0, histN2);

    std::vector<WeightType> histN3(NUM_LABELS, 0);
    histN3[0] = 10;
    histN3[2] = 20;
    boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> > n3 =
            boost::make_shared<RandomTree<PixelInstance, ImageFeatureFunction> >(3, 2, n1, histN3);

    std::vector<WeightType> histN4(NUM_LABELS, 0);
    histN4[1] = 50;
    histN4[2] = 20;
    boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> > n4 =
            boost::make_shared<RandomTree<PixelInstance, ImageFeatureFunction> >(4, 2, n1, histN4);

    size_t featureId1 = 1;
    float threshold1 = 28.391;
    ScoreType score1 = 0.392;
    ImageFeatureFunction feature1(COLOR,
            Offset(-10, 5), Region(7, 3), 1,
            Offset(27, -19), Region(65, 73), 2);
    SplitFunction<PixelInstance, ImageFeatureFunction> split1(featureId1, feature1, threshold1, score1);

    n1->addChildren(split1, n3, n4);

    size_t featureId2 = 2;
    float threshold2 = -29.1245;
    ScoreType score2 = 0.9371;
    ImageFeatureFunction feature2(DEPTH,
            Offset(-18, 25), Region(4, 19), 0,
            Offset(9, 28), Region(1, 16), 0);
    SplitFunction<PixelInstance, ImageFeatureFunction> split2(featureId2, feature2, threshold2, score2);

    n0->addChildren(split2, n1, n2);

    BOOST_CHECK(n0->isRoot());
    BOOST_CHECK_EQUAL(5, n0->countNodes());

    cuv::ndarray<WeightType, cuv::host_memory_space> classLabelPriorDistribution(NUM_LABELS);
    for (size_t i = 0; i < NUM_LABELS; i++) {
        classLabelPriorDistribution[i] = 100;
    }

    boost::shared_ptr<RandomTreeImage> randomTreeImage = boost::make_shared<RandomTreeImage>(n0, configuration,
            classLabelPriorDistribution);

    randomTreeImage->normalizeHistograms(0.0);

    boost::shared_ptr<const TreeNodes> treeData = convertTree(randomTreeImage);
    BOOST_CHECK_EQUAL(n0->countNodes(), static_cast<size_t>(treeData->numNodes()));
    BOOST_CHECK_EQUAL(n0->getNumClasses(), static_cast<size_t>(treeData->numLabels()));

    checkNode(n0, treeData, &split2);
    checkNode(n1, treeData, &split1);
    BOOST_REQUIRE(n2->isLeaf());
    BOOST_REQUIRE(n3->isLeaf());
    BOOST_REQUIRE(n4->isLeaf());
    checkNode(n2, treeData);
    checkNode(n3, treeData);
    checkNode(n4, treeData);

    // do classify

    const size_t treeCacheSize = 3;

    {
        RGBDImage image(640, 480);
        image.calculateIntegral();

        {
            utils::Profile classifyImageTimer("classifyImage");
            cuv::ndarray<float, cuv::dev_memory_space> output(
                    cuv::extents[NUM_LABELS][image.getHeight()][image.getWidth()]);
            cudaSafeCall(cudaMemset(output.ptr(), 0, static_cast<size_t>(output.size() * sizeof(float))));

            classifyImage(treeCacheSize, output, image, NUM_LABELS, treeData);
        }
    }

}

BOOST_AUTO_TEST_CASE(testRecallLargeForest) {

    unsigned int samplesPerImage = 100;
    const int NUM_FEAT = 1000;
    const int NUM_THRESH = 50;

    unsigned int minSampleCount = 32;
    int maxDepth = 15;
    uint16_t boxRadius = 127;
    uint16_t regionSize = 50;
    static const int NUM_THREADS = 1;
    static const int maxImages = 5;
    static const int imageCacheSize = 5;
    unsigned int maxSamplesPerBatch = 100000;
    AccelerationMode accelerationMode = GPU_AND_CPU_COMPARE;

    TrainingConfiguration configuration(SEED, samplesPerImage, NUM_FEAT, minSampleCount, maxDepth, boxRadius,
            regionSize, NUM_THRESH, NUM_THREADS, maxImages, imageCacheSize, maxSamplesPerBatch, accelerationMode);

    std::vector<PixelInstance> samples;

    const size_t NUM_LABELS = 3;

    const int width = 640;
    const int height = 480;
    std::vector<RGBDImage> images(10, RGBDImage(width, height));

    const int NUM_SAMPLES = samplesPerImage * images.size();
    for (int i = 0; i < NUM_SAMPLES; i++) {
        PixelInstance sample(
                &images.at(i / (NUM_SAMPLES / images.size())), // image
                (i / 100) % NUM_LABELS, // label
                Depth((i % 20) / 10.0 + 0.1), // depth
                i % width, // x
                i % height // y
                        );

        samples.push_back(sample);
    }

    // explicitly test a tree that exceeds the maximal number of nodes per layer
    const size_t numNodes[] = { 10, 100, 3 * NODES_PER_TREE_LAYER + 2 };

    Sampler sampler(4711, 0, 1000);
    Sampler typeSampler(4711, 0, 1);
    Sampler channelSampler(4711, 0, 5);
    Sampler offsetSampler(4711, -120, 120);
    Sampler regionSampler(4711, 0, 20);

    for (size_t treeId = 0; treeId < 3; treeId++) {
        boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> > rootNode =
                boost::make_shared<RandomTree<PixelInstance, ImageFeatureFunction> >(treeId, 0, getPointers(samples),
                        NUM_LABELS);

        std::vector<boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> > > nodes;

        std::map<boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> >,
                SplitFunction<PixelInstance, ImageFeatureFunction> > splits;

        nodes.push_back(rootNode);

        boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> > previousNode = rootNode;

        /**
         * creates a degenerated tree with N nodes
         *
         *                   n0
         *                  /  \
         *                 /    \
         *                n1    n2
         *               / \
         *              /   \
         *             n3   n4
         *            /
         *           /
         *          n5
         *         / \
         *        /   \
         *       n6   n7
         *      /
         *     /
         *   ...
         */
        for (size_t nodeId = 1; nodeId < numNodes[treeId]; nodeId += 2) {

            const size_t level = (nodeId + 1) / 2;
            assert(level > 0 && level < numNodes[treeId]);

            boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> > leftNode =
                    boost::make_shared<RandomTree<PixelInstance, ImageFeatureFunction> >(nodeId + treeId, level,
                            getPointers(samples), NUM_LABELS, previousNode);

            boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> > rightNode =
                    boost::make_shared<RandomTree<PixelInstance, ImageFeatureFunction> >(nodeId + 1 + treeId, level,
                            getPointers(samples), NUM_LABELS, previousNode);

            size_t featureId = sampler.getNext();
            float threshold = sampler.getNext() / 200.0 - 100.0;
            ScoreType score = sampler.getNext() / 1000.0;
            assertProbability(score);

            ImageFeatureFunction feature(static_cast<FeatureType>(typeSampler.getNext()),
                    Offset(offsetSampler.getNext(), offsetSampler.getNext()),
                    Region(regionSampler.getNext(), regionSampler.getNext()),
                    channelSampler.getNext(),
                    Offset(offsetSampler.getNext(), offsetSampler.getNext()),
                    Region(regionSampler.getNext(), regionSampler.getNext()),
                    channelSampler.getNext());
            SplitFunction<PixelInstance, ImageFeatureFunction> split(featureId, feature, threshold, score);

            previousNode->addChildren(split, leftNode, rightNode);

            splits[previousNode] = split;

            nodes.push_back(leftNode);
            nodes.push_back(rightNode);

            previousNode = leftNode;
        }

        BOOST_CHECK_EQUAL(treeId, rootNode->getTreeId());
        BOOST_CHECK(rootNode->isRoot());
        BOOST_CHECK_EQUAL(numNodes[treeId] + 1, rootNode->countNodes());

        cuv::ndarray<WeightType, cuv::host_memory_space> classLabelPriorDistribution(NUM_LABELS);
        for (size_t i = 0; i < NUM_LABELS; i++) {
            classLabelPriorDistribution[i] = 100;
        }

        boost::shared_ptr<RandomTreeImage> randomTreeImage =
                boost::make_shared<RandomTreeImage>(rootNode, configuration, classLabelPriorDistribution);

        randomTreeImage->normalizeHistograms(0.0);

        boost::shared_ptr<const TreeNodes> treeData = convertTree(randomTreeImage);
        BOOST_CHECK_EQUAL(rootNode->countNodes(), static_cast<size_t>(treeData->numNodes()));
        BOOST_CHECK_EQUAL(rootNode->getNumClasses(), static_cast<size_t>(treeData->numLabels()));

        CURFIL_INFO("checking nodes");

        assert(nodes.size() == numNodes[treeId] + 1);

        for (size_t nodeId = 0; nodeId < numNodes[treeId]; nodeId++) {
            const boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> > node = nodes[nodeId];
            if (node->isLeaf()) {
                checkNode(node, treeData);
            } else {
                checkNode(node, treeData, &splits[node]);
            }
        }

        CURFIL_INFO("checked " << numNodes[treeId] << " nodes of tree " << treeId);
    }

}
BOOST_AUTO_TEST_SUITE_END()
