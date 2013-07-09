#ifndef CURFIL_RANDOMTREE_H
#define CURFIL_RANDOMTREE_H

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <cassert>
#include <cmath>
#include <cuv/ndarray.hpp>
#include <functional>
#include <limits>
#include <map>
#include <ostream>
#include <set>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <utility>
#include <vector>

#include "score.h"
#include "utils.h"

namespace curfil {

typedef uint8_t LabelType;
typedef unsigned int WeightType;
typedef double FeatureResponseType;

namespace detail {

bool isScoreBetter(const ScoreType bestScore, const ScoreType score, const int featureNr);

cuv::ndarray<double, cuv::host_memory_space> normalizeHistogram(
        const cuv::ndarray<WeightType, cuv::host_memory_space>& histogram,
        const cuv::ndarray<WeightType, cuv::host_memory_space>& priorDistribution,
        const double histogramBias);

}

// Test/training-time classes
enum SplitBranch {
    LEFT, RIGHT
};

enum AccelerationMode {
    CPU_ONLY,
    GPU_ONLY,
    GPU_AND_CPU_COMPARE
};

template<class Instance, class FeatureFunction> class SplitFunction {
public:

    // Note: feat is copied and SplitFunction assumes ownership.
    // To be able to test on any sample, FeatureFunction must be implemented
    // such that it can lookup these Instances dynamically.
    SplitFunction(size_t featureId, const FeatureFunction& feature, float threshold, ScoreType score) :
            featureId(featureId), feature(feature), threshold(threshold), score(score) {
    }

    SplitFunction() :
            featureId(0), feature(), threshold(std::numeric_limits<float>::quiet_NaN()), score(
                    std::numeric_limits<float>::quiet_NaN()) {
    }

    SplitFunction& operator=(const SplitFunction& other) {
        assert(!isnan(other.threshold));
        assert(!isnan(other.score));
        featureId = other.featureId;
        feature = other.feature;
        threshold = other.threshold;
        score = other.score;
        return (*this);
    }

    // Return left or right branch for a given instance and feature function.
    SplitBranch split(const Instance& instance) const {
        return (feature.calculateFeatureResponse(instance) <= getThreshold() ? LEFT : RIGHT);
    }

    // Return the underlying feature used
    const FeatureFunction& getFeature() const {
        return (feature);
    }

    float getThreshold() const {
        assert(!isnan(threshold));
        return threshold;
    }

    ScoreType getScore() const {
        assert(!isnan(score));
        return score;
    }

    size_t getFeatureId() const {
        return featureId;
    }

private:
    size_t featureId;
    FeatureFunction feature;
    float threshold;
    ScoreType score;
};

class TrainingConfiguration {

public:
    TrainingConfiguration(const TrainingConfiguration& other);

    TrainingConfiguration(int randomSeed,
            unsigned int samplesPerImage,
            unsigned int featureCount,
            unsigned int minSampleCount,
            int maxDepth,
            uint16_t boxRadius,
            uint16_t regionSize,
            uint16_t thresholds,
            int numThreads,
            int maxImages,
            int imageCacheSize,
            unsigned int maxSamplesPerBatch,
            AccelerationMode accelerationMode,
            bool useCIELab = true,
            bool useDepthFilling = false,
            const std::vector<int> deviceIds = std::vector<int>(1, 0),
            const std::string subsamplingType = "classUniform",
            const std::vector<std::string>& ignoredColors = std::vector<std::string>()) :
            randomSeed(randomSeed),
                    samplesPerImage(samplesPerImage),
                    featureCount(featureCount),
                    minSampleCount(minSampleCount),
                    maxDepth(maxDepth),
                    boxRadius(boxRadius),
                    regionSize(regionSize),
                    thresholds(thresholds),
                    numThreads(numThreads),
                    maxImages(maxImages),
                    imageCacheSize(imageCacheSize),
                    maxSamplesPerBatch(maxSamplesPerBatch),
                    accelerationMode(accelerationMode),
                    useCIELab(useCIELab),
                    useDepthFilling(useDepthFilling),
                    deviceIds(deviceIds),
                    subsamplingType(subsamplingType),
                    ignoredColors(ignoredColors)
    {
        for (size_t c = 0; c < ignoredColors.size(); c++) {
            if (ignoredColors[c].empty()) {
                throw std::runtime_error(std::string("illegal color: '") + ignoredColors[c] + "'");
            }
        }
        if (maxImages > 0 && maxImages < imageCacheSize) {
            throw std::runtime_error(
                    (boost::format("illegal configuration: maxImages (%d) must not be lower than imageCacheSize (%d)")
                            % maxImages % imageCacheSize).str());
        }
    }

    void setRandomSeed(int randomSeed) {
        this->randomSeed = randomSeed;
    }

    int getRandomSeed() const {
        return randomSeed;
    }

    unsigned int getSamplesPerImage() const {
        return samplesPerImage;
    }

    unsigned int getFeatureCount() const {
        return featureCount;
    }

    unsigned int getMinSampleCount() const {
        return minSampleCount;
    }

    int getMaxDepth() const {
        return maxDepth;
    }

    uint16_t getBoxRadius() const {
        return boxRadius;
    }

    uint16_t getRegionSize() const {
        return regionSize;
    }

    uint16_t getThresholds() const {
        return thresholds;
    }

    int getNumThreads() const {
        return numThreads;
    }

    int getMaxImages() const {
        return maxImages;
    }

    int getImageCacheSize() const {
        return imageCacheSize;
    }

    unsigned int getMaxSamplesPerBatch() const {
        return maxSamplesPerBatch;
    }

    AccelerationMode getAccelerationMode() const {
        return accelerationMode;
    }

    void setAccelerationMode(const AccelerationMode& accelerationMode) {
        this->accelerationMode = accelerationMode;
    }

    static AccelerationMode parseAccelerationModeString(const std::string& modeString);

    std::string getAccelerationModeString() const;

    const std::vector<int>& getDeviceIds() const {
        return deviceIds;
    }

    void setDeviceIds(const std::vector<int>& deviceIds) {
        this->deviceIds = deviceIds;
    }

    std::string getSubsamplingType() const {
        return subsamplingType;
    }

    bool isUseCIELab() const {
        return useCIELab;
    }

    bool isUseDepthFilling() const {
        return useDepthFilling;
    }

    const std::vector<std::string>& getIgnoredColors() const {
        return ignoredColors;
    }

    TrainingConfiguration& operator=(const TrainingConfiguration& other);

    bool equals(const TrainingConfiguration& other, bool strict = false) const;

    bool operator==(const TrainingConfiguration& other) const {
        return this->equals(other, true);
    }

    bool operator!=(const TrainingConfiguration& other) const {
        return (!(*this == other));
    }

private:

    // do not forget to update operator==/operator= as well!
    int randomSeed;
    unsigned int samplesPerImage;
    unsigned int featureCount;
    unsigned int minSampleCount;
    int maxDepth;
    uint16_t boxRadius;
    uint16_t regionSize;
    uint16_t thresholds;
    int numThreads;
    int maxImages;
    int imageCacheSize;
    unsigned int maxSamplesPerBatch;
    AccelerationMode accelerationMode;
    bool useCIELab;
    bool useDepthFilling;
    std::vector<int> deviceIds;
    std::string subsamplingType;
    std::vector<std::string> ignoredColors;
};

template<class Instance, class FeatureFunction>
class RandomTree {

public:

    RandomTree(const size_t& nodeId, const int level,
            const std::vector<const Instance*>& samples, size_t numClasses,
            const boost::shared_ptr<RandomTree<Instance, FeatureFunction> >& parent = boost::shared_ptr<
                    RandomTree<Instance, FeatureFunction> >()) :
            nodeId(nodeId), level(level), parent(parent), leaf(true), trainSamples(),
                    numClasses(numClasses), histogram(numClasses), timers(),
                    split(), left(), right() {

        assert(histogram.ndim() == 1);
        for (size_t label = 0; label < numClasses; label++) {
            histogram[label] = 0;
        }

        for (size_t i = 0; i < samples.size(); i++) {
            histogram[samples[i]->getLabel()] += samples[i]->getWeight();
            trainSamples.push_back(*samples[i]);
        }
    }

    RandomTree(const size_t& nodeId, const int level,
            const boost::shared_ptr<RandomTree<Instance, FeatureFunction> >& parent,
            const std::vector<WeightType>& histogram) :
            nodeId(nodeId), level(level), parent(parent), leaf(true), trainSamples(),
                    numClasses(histogram.size()), histogram(histogram.size()), timers(),
                    split(), left(), right() {

        for (size_t i = 0; i < histogram.size(); i++) {
            this->histogram[i] = histogram[i];
        }

    }

    /**
     * returns true iff the entropy is zero
     */
    bool hasPureHistogram() const {
        size_t nonZeroClasses = 0;
        for (size_t i = 0; i < histogram.size(); i++) {
            const WeightType classCount = histogram[i];
            if (classCount == 0) {
                continue;
            }
            nonZeroClasses++;
            if (nonZeroClasses > 1) {
                return false;
            }
        }
        assert(nonZeroClasses > 0);
        return (nonZeroClasses == 1);
    }

    size_t getNumClasses() const {
        assert(histogram.size() == numClasses);
        return numClasses;
    }

    // For a given instance, collect the set of all nodes this sample
    // traverses through
    void collectNodeIndices(const Instance& instance,
            std::set<unsigned int>& nodeSet, bool includeRoot) const {

        if (!isRoot() || includeRoot) {
            nodeSet.insert(nodeId);
        }
        if (isLeaf())
            return;

        boost::shared_ptr<RandomTree<Instance, FeatureFunction> > node;
        if (split.split(instance) == LEFT) {
            node = left;
        } else {
            node = right;
        }
        assert(node != NULL);
        node->collectNodeIndices(instance, nodeSet, includeRoot);
    }

    // Classify an instance by traversing the tree and returning the tree leaf
    // nodes leaf class.
    LabelType classify(const Instance& instance) const {
        return traverseToLeaf(instance)->getDominantClass();
    }

    const cuv::ndarray<double, cuv::host_memory_space>& classifySoft(const Instance& instance) const {
        return (traverseToLeaf(instance)->getNormalizedHistogram());
    }

    size_t getNumTrainSamples() const {
        return trainSamples.size();
    }

    const std::map<std::string, double>& getTimerValues() const {
        return timers;
    }

    const std::map<std::string, std::string>& getTimerAnnotations() const {
        return timerAnnotations;
    }

    void setTimerValue(const std::string& key, utils::Timer& timer) {
        setTimerValue(key, timer.getSeconds());
    }

    template<class V>
    void setTimerAnnotation(const std::string& key, const V& annotation) {
        timerAnnotations[key] = boost::lexical_cast<std::string>(annotation);
    }

    void addTimerValue(const std::string& key, utils::Timer& timer) {
        addTimerValue(key, timer.getSeconds());
    }

    void setTimerValue(const std::string& key, const double timeInSeconds) {
        timers[key] = timeInSeconds;
    }

    void addTimerValue(const std::string& key, const double timeInSeconds) {
        timers[key] += timeInSeconds;
    }

    boost::shared_ptr<const RandomTree<Instance, FeatureFunction> > getLeft() const {
        return left;
    }

    boost::shared_ptr<const RandomTree<Instance, FeatureFunction> > getRight() const {
        return right;
    }

    int getLevel() const {
        return level;
    }

    bool isRoot() const {
        return (parent.lock().get() == NULL);
    }

    const RandomTree<Instance, FeatureFunction>* getRoot() const {
        if (isRoot()) {
            return this;
        }
        return parent.lock()->getRoot();
    }

    void countFeatures(std::map<std::string, size_t>& featureCounts) const {
        if (isLeaf()) {
            return;
        }
        std::string featureType(split.getFeature().getTypeString());
        featureCounts[featureType]++;

        left->countFeatures(featureCounts);
        right->countFeatures(featureCounts);
    }

    size_t countNodes() const {
        return (isLeaf() ? 1 : (1 + left->countNodes() + right->countNodes()));
    }

    size_t countLeafNodes() const {
        return (isLeaf() ? 1 : (left->countLeafNodes() + right->countLeafNodes()));
    }

    size_t getTreeDepth() const {
        if (isLeaf())
            return 1;
        return (1 + std::max(left->getTreeDepth(), right->getTreeDepth()));
    }

    // Links the given left/right subtrees as children to this one.
    // Assigns unique labels to the children nodes.
    // Makes the current node a non-leaf node.
    // This tree node assumes ownership of split and both children.
    void addChildren(const SplitFunction<Instance, FeatureFunction>& split,
            boost::shared_ptr<RandomTree<Instance, FeatureFunction> > left,
            boost::shared_ptr<RandomTree<Instance, FeatureFunction> > right) {
        assert(isLeaf());

        assert(left.get());
        assert(right.get());

        assert(left->getNodeId() > this->getNodeId());
        assert(right->getNodeId() > this->getNodeId());

        assert(left->parent.lock().get() == this);
        assert(right->parent.lock().get() == this);

        assert(this->left.get() == 0);
        assert(this->right.get() == 0);

        // Link
        this->left = left;
        this->right = right;
        this->split = split;

        // This node becomes interior to the tree
        leaf = false;
    }

    bool isLeaf() const {
        return leaf;
    }

    const SplitFunction<Instance, FeatureFunction>& getSplit() const {
        return split;
    }

    size_t getNodeId() const {
        return nodeId;
    }

    size_t getTreeId() const {
        size_t rootNodeId = getRoot()->getNodeId();
        assert(rootNodeId <= getNodeId());
        return rootNodeId;
    }

    void normalizeHistograms(const cuv::ndarray<WeightType, cuv::host_memory_space>& priorDistribution,
            const double histogramBias) {

        normalizedHistogram = detail::normalizeHistogram(histogram, priorDistribution, histogramBias);
        assert(normalizedHistogram.shape() == histogram.shape());

        if (isLeaf()) {
            double sum = 0;
            for (size_t i = 0; i < normalizedHistogram.size(); i++) {
                sum += normalizedHistogram[i];
            }
            if (sum == 0) {
                INFO("normalized histogram of node " << getNodeId() << ", level " << getLevel() << " has zero sum");
            }
        } else {
            left->normalizeHistograms(priorDistribution, histogramBias);
            right->normalizeHistograms(priorDistribution, histogramBias);
        }

        if (normalizedHistogram.shape() != histogram.shape()) {
            ERROR("node: " << nodeId << " (level " << level << ")");
            ERROR("histogram: " << histogram);
            ERROR("normalized histogram: " << normalizedHistogram);
            throw std::runtime_error("failed to normalize histogram");
        }
    }

    const cuv::ndarray<WeightType, cuv::host_memory_space>& getHistogram() const {
        return histogram;
    }

    const cuv::ndarray<double, cuv::host_memory_space>& getNormalizedHistogram() const {
        if (normalizedHistogram.shape() != histogram.shape()) {
            ERROR("node: " << nodeId << " (level " << level << ")");
            ERROR("histogram: " << histogram);
            ERROR("normalized histogram: " << normalizedHistogram);
            throw std::runtime_error("histogram not normalized");
        }
        return normalizedHistogram;
    }

    const std::vector<Instance>& getTrainSamples() const {
        return trainSamples;
    }

private:
    // A unique node identifier within this tree
    const size_t nodeId;
    const int level;

    /*
     * Parent node or NULL if this is the root
     *
     * Having a weak pointer here is important to avoid loops which cause a memory leak
     */
    const boost::weak_ptr<RandomTree<Instance, FeatureFunction> > parent;

    // If true, this node is a leaf node
    bool leaf;

    std::vector<Instance> trainSamples;

    size_t numClasses;

    cuv::ndarray<WeightType, cuv::host_memory_space> histogram;

    cuv::ndarray<double, cuv::host_memory_space> normalizedHistogram;

    std::map<std::string, double> timers;
    std::map<std::string, std::string> timerAnnotations;

    // If isLeaf is false, split and both left and right must be non-NULL
    // If isLeaf is true, split = left = right = NULL
    SplitFunction<Instance, FeatureFunction> split;
    boost::shared_ptr<RandomTree<Instance, FeatureFunction> > left;
    boost::shared_ptr<RandomTree<Instance, FeatureFunction> > right;

    LabelType getDominantClass() const {
        double max = std::numeric_limits<double>::quiet_NaN();
        assert(histogram.size() == numClasses);
        LabelType maxClass = 0;

        for (LabelType classNr = 0; classNr < histogram.size(); classNr++) {
            const WeightType& count = histogram[classNr];
            assert(count >= 0.0);
            if (isnan(max) || count > max) {
                max = count;
                maxClass = classNr;
            }
        }
        return maxClass;
    }

    const RandomTree<Instance, FeatureFunction>* traverseToLeaf(const Instance& instance) const {
        if (isLeaf())
            return this;

        assert(left.get());
        assert(right.get());

        if (split.split(instance) == LEFT) {
            return left->traverseToLeaf(instance);
        } else {
            return right->traverseToLeaf(instance);
        }
    }

};

class Sampler {
public:
    Sampler(int seed, int lower, int upper) :
            seed(seed), lower(lower), upper(upper), rng(seed), distribution(lower, upper) {
        assert(upper >= lower);
    }

    Sampler(const Sampler& other) :
            seed(other.seed), lower(other.lower), upper(other.upper), rng(seed), distribution(lower, upper) {
    }

    int getNext();

    int getSeed() const {
        return seed;
    }

    int getLower() const {
        return upper;
    }

    int getUpper() const {
        return upper;
    }

private:
    Sampler& operator=(const Sampler&);
    Sampler();

    int seed;
    int lower;
    int upper;

    boost::mt19937 rng;
    boost::uniform_int<> distribution;
};

template<class T>
class ReservoirSampler {
public:

    ReservoirSampler() :
            count(0), samples(0), reservoir() {
    }

    ReservoirSampler(size_t samples) :
            count(0), samples(samples), reservoir() {
        reservoir.reserve(samples);
        assert(reservoir.empty());
    }

    void sample(Sampler& sampler, const T& sample) {

        if (samples == 0) {
            throw std::runtime_error("no samples to sample");
        }

        assert(sampler.getUpper() > static_cast<int>(count));

        if (reservoir.size() < samples) {
            reservoir.push_back(sample);
        } else {
            assert(count >= samples);
            // draw a random number from the interval (0, count) including count!
            size_t rand = sampler.getNext() % (count + 1);
            if (rand < samples) {
                reservoir[rand] = sample;
            }
        }

        count++;
    }

    const std::vector<T>& getReservoir() const {
        return reservoir;
    }

private:
    size_t count;
    size_t samples;
    std::vector<T> reservoir;
};

// Training-time classes
class RandomSource {

private:
    int seed;

public:
    RandomSource(int seed) :
            seed(seed) {
    }

    Sampler uniformSampler(int val) {
        return uniformSampler(0, val - 1);
    }

    Sampler uniformSampler(int lower, int upper) {
        return Sampler(seed++, lower, upper);
    }
};

template<class Instance, class FeatureEvaluation, class FeatureFunction>
class RandomTreeTrain {
public:
    /* featureCount: K >= 1, the number of random splits to sample and evaluate
     *    at each interior node.  Note that K=1 yields completely randomized
     *    trees, whereas large values of K lead to aggressive optimization of
     *    the split.  A common choice is round(sqrt(N)), where N is the number
     *    of sample instances in the training set passed to the Train method.
     * minSampleCount: n_min, the minimum number of training samples
     *    necessary such that training is continued.
     * maxTreeDepth: if < 0, no limit on the tree depth is assumed.  If > 0,
     *    the tree learning is stopped when the given depth is reached.
     */
    RandomTreeTrain(int id, size_t numClasses, const TrainingConfiguration& configuration) :
            id(id), numClasses(numClasses), configuration(configuration) {
        assert(configuration.getMaxDepth() > 0);
    }

private:

    bool shouldContinueGrowing(const boost::shared_ptr<RandomTree<Instance, FeatureFunction> > node) const {
        if (node->hasPureHistogram()) {
            return false;
        }
        if (node->getNumTrainSamples() < configuration.getMinSampleCount()) {
            return false;
        }

        return true;
    }

    typedef boost::shared_ptr<RandomTree<Instance, FeatureFunction> > RandomTreePointer;
    typedef std::vector<const Instance*> Samples;

    void compareHistograms(boost::shared_ptr<RandomTree<Instance, FeatureFunction> >& currentNode,
            boost::shared_ptr<RandomTree<Instance, FeatureFunction> >& leftNode,
            boost::shared_ptr<RandomTree<Instance, FeatureFunction> >& rightNode,
            const SplitFunction<Instance, FeatureFunction>& bestSplit) const {

        size_t size = currentNode->getHistogram().size();
        WeightType leftHistogram[size];
        WeightType rightHistogram[size];
        const unsigned int leftRightStride = 1; // consecutive in memory
        WeightType allHistogram[size];
        for (size_t i = 0; i < size; i++) {
            leftHistogram[i] = leftNode->getHistogram()[i];
            rightHistogram[i] = rightNode->getHistogram()[i];
            allHistogram[i] = currentNode->getHistogram()[i];
        }
        // assert that actual score is the same as the calculated score
        double totalLeft = sum(leftNode->getHistogram());
        double totalRight = sum(rightNode->getHistogram());
        double actualScore = NormalizedInformationGainScore::calculateScore(size, leftHistogram, rightHistogram,
                leftRightStride,
                allHistogram, totalLeft, totalRight);
        double diff = std::fabs(actualScore - bestSplit.getScore());
        if (diff > 0.02) {
            std::ostringstream o;
            o.precision(10);
            o << "actual score and best split score differ: " << diff << std::endl;
            o << "actual score: " << actualScore << std::endl;
            o << "best split score: " << bestSplit.getScore() << std::endl;
            o << "total left: " << totalLeft << std::endl;
            o << "total right: " << totalRight << std::endl;
            o << "histogram:  " << currentNode->getHistogram() << std::endl;
            o << "histogram left:  " << leftNode->getHistogram() << std::endl;
            o << "histogram right: " << rightNode->getHistogram() << std::endl;
            throw std::runtime_error(o.str());
        }
    }

public:

    /* Train a single random tree breadth-first */
    void train(FeatureEvaluation& featureEvaluation,
            RandomSource& randomSource,
            const std::vector<std::pair<RandomTreePointer, Samples> >& samplesPerNode,
            int idNode, int currentLevel = 1) const {

        // Depth exhausted: leaf node
        if (currentLevel == configuration.getMaxDepth()) {
            return;
        }

        INFO("training level " << currentLevel << ". nodes: " << samplesPerNode.size());

        utils::Timer trainTimer;

        std::vector<std::pair<RandomTreePointer, Samples> > samplesPerNodeNextLevel;

        std::vector<SplitFunction<Instance, FeatureFunction> > bestSplits = featureEvaluation.evaluateBestSplits(
                randomSource, samplesPerNode);

        assert(bestSplits.size() == samplesPerNode.size());

        for (size_t i = 0; i < samplesPerNode.size(); i++) {

            const std::pair<RandomTreePointer, Samples>& it = samplesPerNode[i];

            const std::vector<const Instance*>& samples = it.second;

            boost::shared_ptr<RandomTree<Instance, FeatureFunction> > currentNode = it.first;
            assert(currentNode);

            const SplitFunction<Instance, FeatureFunction>& bestSplit = bestSplits[i];

            // Split all training instances into the subtrees.  The way we train
            // the trees ensures we have no more than 2*N instances to manage in
            // memory.
            std::vector<const Instance*> samplesLeft;
            std::vector<const Instance*> samplesRight;

            for (size_t sample = 0; sample < samples.size(); sample++) {
                assert(samples[sample] != NULL);
                if (bestSplit.split(*samples[sample]) == LEFT) {
                    samplesLeft.push_back(samples[sample]);
                } else {
                    samplesRight.push_back(samples[sample]);
                }
            }

            assert(samplesLeft.size() + samplesRight.size() == samples.size());

            boost::shared_ptr<RandomTree<Instance, FeatureFunction> > leftNode = boost::make_shared<RandomTree<Instance,
                    FeatureFunction> >(++idNode, currentNode->getLevel() + 1, samplesLeft, numClasses, currentNode);

            boost::shared_ptr<RandomTree<Instance, FeatureFunction> > rightNode = boost::make_shared<
                    RandomTree<Instance, FeatureFunction> >(++idNode, currentNode->getLevel() + 1, samplesRight,
                    numClasses, currentNode);

#ifndef NDEBUG
            compareHistograms(currentNode, leftNode, rightNode, bestSplit);
#endif

            if (samplesLeft.empty() || samplesRight.empty()) {
                ERROR("best split score: " << bestSplit.getScore());
                ERROR("samples: " << samples.size());
                ERROR("threshold: " << bestSplit.getThreshold());
                ERROR("feature: " << bestSplit.getFeature());
                ERROR("histogram: " << currentNode->getHistogram());
                ERROR("samplesLeft: " << samplesLeft.size());
                ERROR("samplesRight: " << samplesRight.size());

                compareHistograms(currentNode, leftNode, rightNode, bestSplit);

                if (samplesLeft.empty()) {
                    throw std::runtime_error("no samples in left node");
                }
                if (samplesRight.empty()) {
                    throw std::runtime_error("no samples in right node");
                }
            }

            currentNode->addChildren(bestSplit, leftNode, rightNode);

            if (shouldContinueGrowing(leftNode)) {
                samplesPerNodeNextLevel.push_back(std::make_pair(leftNode, samplesLeft));
            }

            if (shouldContinueGrowing(rightNode)) {
                samplesPerNodeNextLevel.push_back(std::make_pair(rightNode, samplesRight));
            }
        }

        INFO("training level " << currentLevel << " took " << trainTimer.format(3));
        if (!samplesPerNodeNextLevel.empty()) {
            train(featureEvaluation, randomSource, samplesPerNodeNextLevel, idNode, currentLevel + 1);
        }
    }

private:

    template<class T>
    T sum(const cuv::ndarray<T, cuv::host_memory_space>& vector, const T initial = static_cast<T>(0)) const {
        T v = initial;
        for (size_t i = 0; i < vector.size(); i++) {
            v += vector[i];
        }
        return v;
    }

    int id;
    size_t numClasses;
    const TrainingConfiguration configuration;
}
;

}

std::ostream& operator<<(std::ostream& os, const curfil::TrainingConfiguration& configuration);

#endif
