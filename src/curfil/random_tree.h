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

/**
 * Class used to get the splitting score for a given feature and threshold
 * @ingroup training_helper_classes
 */
template<class Instance, class FeatureFunction> class SplitFunction {
public:

    /**
    * creates a SplitFunction object using the feature and threshold used when splitting and the resulting score
    */
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

    /**
     * uses another SplitFunction object to set the attributes
     */
    SplitFunction& operator=(const SplitFunction& other) {
        assert(!isnan(other.threshold));
        assert(!isnan(other.score));
        featureId = other.featureId;
        feature = other.feature;
        threshold = other.threshold;
        score = other.score;
        return (*this);
    }
        /**
	 * @return left or right branch for a given instance and feature function.
	 */
	SplitBranch split(const Instance& instance, bool& flippedSameSplit) const {
		bool value1 = feature.calculateFeatureResponse(instance) <= getThreshold();
		flippedSameSplit = true;
		bool flipSetting = instance.getFlipping();
		if (flipSetting) {
			bool value2 = feature.calculateFeatureResponse(instance, true) <= getThreshold();
			if (value1 != value2)
				flippedSameSplit = false;
		}
		return ((value1) ? LEFT : RIGHT);
	}

    /**
     * @return the underlying feature used
     */
    const FeatureFunction& getFeature() const {
        return (feature);
    }

    /**
     * @return the threshold used
     */
    float getThreshold() const {
        assert(!isnan(threshold));
        return threshold;
    }

    /**
     * @return the score used
     */
    ScoreType getScore() const {
        assert(!isnan(score));
        return score;
    }

    /**
     * @return the featue ID
     */
    size_t getFeatureId() const {
        return featureId;
    }

private:
    size_t featureId;
    FeatureFunction feature;
    float threshold;
    ScoreType score;
};

/**
 * Parameters used for training, most of which are provided by the user
 * @ingroup training_helper_classes
 */
class TrainingConfiguration {

public:

    explicit TrainingConfiguration() :
            randomSeed(0),
                    samplesPerImage(0),
                    featureCount(0),
                    minSampleCount(0),
                    maxDepth(0),
                    boxRadius(0),
                    regionSize(0),
                    thresholds(0),
                    numThreads(0),
                    maxImages(0),
                    imageCacheSize(0),
                    maxSamplesPerBatch(0),
                    accelerationMode(GPU_ONLY),
                    useCIELab(0),
                    useDepthFilling(0),
                    deviceIds(),
                    subsamplingType(),
                    ignoredColors(),
                    useDepthImages(0),
                    horizontalFlipping(0) {
    }

    /**
     * using another object to set the training cofiguration attributes
     */
    TrainingConfiguration(const TrainingConfiguration& other);

    /**
     * creating a configuration objects with the settings that the user provided
     */
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
            const std::vector<std::string>& ignoredColors = std::vector<std::string>(),
            bool useDepthImages = true,
            bool horizontalFlipping = false) :
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
                    ignoredColors(ignoredColors),
                    useDepthImages(useDepthImages),
                    horizontalFlipping(horizontalFlipping)
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

    /**
     * sets the seed used for RandomSource during training
     */
    void setRandomSeed(int randomSeed) {
        this->randomSeed = randomSeed;
    }

    /**
     * @return the random seed used
     */ 
    int getRandomSeed() const {
        return randomSeed;
    }

    /**
     * @return number of pixels to sample per image
     */
    unsigned int getSamplesPerImage() const {
        return samplesPerImage;
    }

    /**
     * @return number of random feature candidates to sample
     */
    unsigned int getFeatureCount() const {
        return featureCount;
    }

    /**
     * @return minimum number of samples that a node should have to continue splitting
     */
    unsigned int getMinSampleCount() const {
        return minSampleCount;
    }

    /**
     * @return maximum depth of the tree, training is stopped after that
     */
    int getMaxDepth() const {
        return maxDepth;
    }

    /**
     * @return the interval from which to sample the offsets of the rectangular regions
     */
    uint16_t getBoxRadius() const {
        return boxRadius;
    }

    /**
     * @return the interval from which to sample the width and height of the rectangular regions
     */
    uint16_t getRegionSize() const {
        return regionSize;
    }

    /**
     * @return number of threshold candidates selected for the evaluation split
     */
    uint16_t getThresholds() const {
        return thresholds;
    }

    /**
     * @return number of threads used in training
     */
    int getNumThreads() const {
        return numThreads;
    }

    /**
     * @return number of images to load when training each tree
     */
    int getMaxImages() const {
        return maxImages;
    }

    /**
     * @return image cache size on the GPU in Megabytes
     */
    int getImageCacheSize() const {
        return imageCacheSize;
    }

    /**
     * @return max number of samples in a batch - depends on the  memory and sample size
     */
    unsigned int getMaxSamplesPerBatch() const {
        return maxSamplesPerBatch;
    }

    /**
     * @return acceleration mode: gpu, cpu or compare
     */
    AccelerationMode getAccelerationMode() const {
        return accelerationMode;
    }

    /**
     * set the acceleration mode: gpu or cpu
     */
    void setAccelerationMode(const AccelerationMode& accelerationMode) {
        this->accelerationMode = accelerationMode;
    }

    /**
     * @return the acceleration mode from the string representation
     */
    static AccelerationMode parseAccelerationModeString(const std::string& modeString);

    /**
     * @return acceleration mode as string
     */
    std::string getAccelerationModeString() const;

    /**
     * @return GPU deviceId used for training
     */
    const std::vector<int>& getDeviceIds() const {
        return deviceIds;
    }

    /**
     * set the deviceId used for training
     */
    void setDeviceIds(const std::vector<int>& deviceIds) {
        this->deviceIds = deviceIds;
    }

    /**
     * @return type of sampling, pixelUniform or classUniform
     */
    std::string getSubsamplingType() const {
        return subsamplingType;
    }

    /**
     * @return whether to convert images to CIELab color space before training
     */
    bool isUseCIELab() const {
        return useCIELab;
    }

    /**
     * @return whether to do simple depth filling
     */
    bool isUseDepthFilling() const {
        return useDepthFilling;
    }

    /**
     * @return whether there are depth images
     */
    bool isUseDepthImages() const {
    	return useDepthImages;
    }

    /**
     * @return whether data should be augmented with horizontally flipped images
     */
    bool doHorizontalFlipping() const {
    	return horizontalFlipping;
    }

    /**
     * @return which colors should be ignored when sampling
     */
    const std::vector<std::string>& getIgnoredColors() const {
        return ignoredColors;
    }

    /**
     * set its attributes to be equal to another configuration
     */
    TrainingConfiguration& operator=(const TrainingConfiguration& other);

    /**
     * @return whether it's equal to another configuration
     */
    bool equals(const TrainingConfiguration& other, bool strict = false) const;

    /**
     * @return whether it's equal to another configuration, all attributes should match 
     */
    bool operator==(const TrainingConfiguration& other) const {
        return this->equals(other, true);
    }

    /**
     * @return whether it's not equal to another configuration
     */
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
    bool useDepthImages;
    bool horizontalFlipping;
};

/**
 * A sub-tree with histograms and pointers to the parent and children
 * @ingroup forest_hierarchy
 */
template<class Instance, class FeatureFunction>
class RandomTree {

public:

    /**
     * create a sub-tree at the given level and nodeId using samples provided
     */
    RandomTree(const size_t& nodeId, const int level,
            const std::vector<const Instance*>& samples, size_t numClasses,
            const boost::shared_ptr<RandomTree<Instance, FeatureFunction> >& parent = boost::shared_ptr<
                    RandomTree<Instance, FeatureFunction> >()) :
            nodeId(nodeId), level(level), parent(parent), leaf(true), trainSamples(),
                    numClasses(numClasses), histogram(numClasses), allPixelsHistogram(numClasses), timers(),
                    split(), left(), right() {

        assert(histogram.ndim() == 1);
        for (size_t label = 0; label < numClasses; label++) {
            histogram[label] = 0;
            allPixelsHistogram[label] = 0;
        }

        for (size_t i = 0; i < samples.size(); i++) {
            histogram[samples[i]->getLabel()] += samples[i]->getWeight();
            trainSamples.push_back(*samples[i]);
            if (samples[i]->getFlipping() == true) {
            	histogram[samples[i]->getLabel()] += samples[i]->getWeight();
            }
        }
    }

    /**
     * create a sub-tree at the given level and nodeId using the histogram provided
     */ 
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

    /**
     * @return the number of labels
     */
    size_t getNumClasses() const {
        assert(histogram.size() == numClasses);
        return numClasses;
    }

    /** 
     * For a given instance, collect the set of all nodes this sample traverses through
     */
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
 
    /**
     * collect all leaf nodes
     */
    void collectLeafNodes(std::vector<size_t> &leafSet) {
     	if (isLeaf())
     	{
         	leafSet.push_back(this->getNodeId());
     	}
         else
         {
         	left->collectLeafNodes(leafSet);
         	right->collectLeafNodes(leafSet);
         }
     }

    /**
     * Classify an instance by traversing the tree and returning the tree leaf nodes leaf class.
     */
    LabelType classify(const Instance& instance) const {
        return traverseToLeaf(instance)->getDominantClass();
    }

    /**
     * @return the histogram of the leaf reached when classifying the instance
     */
    const cuv::ndarray<double, cuv::host_memory_space>& classifySoft(const Instance& instance) const {
        return (traverseToLeaf(instance)->getNormalizedHistogram());
    }

    /**
     * @return number of samples
     */
    size_t getNumTrainSamples() const {
        return trainSamples.size();
    }

    /**
     * @return the timers for the subtree
     */
    const std::map<std::string, double>& getTimerValues() const {
        return timers;
    }

    /** 
     * @return the timers annotations
     */
    const std::map<std::string, std::string>& getTimerAnnotations() const {
        return timerAnnotations;
    }

    /**
     * set the timer value used when profiling
     */
    void setTimerValue(const std::string& key, utils::Timer& timer) {
        setTimerValue(key, timer.getSeconds());
    }

    /**
     * set the timer annotation
     */ 
    template<class V>
    void setTimerAnnotation(const std::string& key, const V& annotation) {
        timerAnnotations[key] = boost::lexical_cast<std::string>(annotation);
    }
 
    /**
     * add the given timer seconds
     */
    void addTimerValue(const std::string& key, utils::Timer& timer) {
        addTimerValue(key, timer.getSeconds());
    }

    /**
     * set a timer to the given seconds
     */
    void setTimerValue(const std::string& key, const double timeInSeconds) {
        timers[key] = timeInSeconds;
    }

    /**
     * add the given seconds to the timer
     */
    void addTimerValue(const std::string& key, const double timeInSeconds) {
        timers[key] += timeInSeconds;
    }

    /**
     * @return the left branch of the tree
     */
    boost::shared_ptr<const RandomTree<Instance, FeatureFunction> > getLeft() const {
        return left;
    }

    /**
     * @return the right branch of the treeunit_testing
     */
    boost::shared_ptr<const RandomTree<Instance, FeatureFunction> > getRight() const {
        return right;
    }

    /**
     * @return the current level
     */
    int getLevel() const {
        return level;
    }

    /**
     * @return whether this is the root node
     */
    bool isRoot() const {
        return (parent.lock().get() == NULL);
    }

    /**
     * @return the root of the tree
     */
    const RandomTree<Instance, FeatureFunction>* getRoot() const {
        if (isRoot()) {
            return this;
        }
        return parent.lock()->getRoot();
    }

    /**
     * @return the number of features grouped by their type
     */
    void countFeatures(std::map<std::string, size_t>& featureCounts) const {
        if (isLeaf()) {
            return;
        }
        std::string featureType(split.getFeature().getTypeString());
        featureCounts[featureType]++;

        left->countFeatures(featureCounts);
        right->countFeatures(featureCounts);
    }

    /**
     * @return the number of nodes
     */
    size_t countNodes() const {
        return (isLeaf() ? 1 : (1 + left->countNodes() + right->countNodes()));
    }

    /**
     * @return number of leaf nodes
     */
    size_t countLeafNodes() const {
        return (isLeaf() ? 1 : (left->countLeafNodes() + right->countLeafNodes()));
    }

    /**
     * @return depth of the tree
     */
    size_t getTreeDepth() const {
        if (isLeaf())
            return 1;
        return (1 + std::max(left->getTreeDepth(), right->getTreeDepth()));
    }

    /**
     * Links the given left/right subtrees as children to this one.
     * 
     * Assigns unique labels to the children nodes.
     * Makes the current node a non-leaf node.
     * This tree node assumes ownership of split and both children.
     */
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

    /**
     * @return whether this is a leaf node
     */
    bool isLeaf() const {
        return leaf;
    }

    /**
     * @return the split associated with the subtree
     */
    const SplitFunction<Instance, FeatureFunction>& getSplit() const {
        return split;
    }

    /**
     * @return the current node Id
     */
    size_t getNodeId() const {
        return nodeId;
    }
   
    /**
     * @return the tree Id
     */
    size_t getTreeId() const {
        size_t rootNodeId = getRoot()->getNodeId();
        assert(rootNodeId <= getNodeId());
        return rootNodeId;
    }

    /**
     * normalize the histograms of the tree and its branches
     */
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
                CURFIL_INFO("normalized histogram of node " << getNodeId() << ", level " << getLevel() << " has zero sum");
            }
        } else {
            left->normalizeHistograms(priorDistribution, histogramBias);
            right->normalizeHistograms(priorDistribution, histogramBias);
        }

        if (normalizedHistogram.shape() != histogram.shape()) {
            CURFIL_ERROR("node: " << nodeId << " (level " << level << ")");
            CURFIL_ERROR("histogram: " << histogram);
            CURFIL_ERROR("normalized histogram: " << normalizedHistogram);
            throw std::runtime_error("failed to normalize histogram");
        }
    }

    /**
     * @return the histogram associated with the node
     */
    const cuv::ndarray<WeightType, cuv::host_memory_space>& getHistogram() const {
        return histogram;
    }

    /**
     * @return the normalized histogram associated with the node
     */
    const cuv::ndarray<double, cuv::host_memory_space>& getNormalizedHistogram() const {
        if (normalizedHistogram.shape() != histogram.shape()) {
            CURFIL_ERROR("node: " << nodeId << " (level " << level << ")");
            CURFIL_ERROR("histogram: " << histogram);
            CURFIL_ERROR("normalized histogram: " << normalizedHistogram);
            throw std::runtime_error("histogram not normalized");
        }
        return normalizedHistogram;
    }

    /**
     * @return the train samples for the current node
     */
    const std::vector<Instance>& getTrainSamples() const {
        return trainSamples;
    }

    /**
     * add the passed value to the label histogram count
     */
    void setAllPixelsHistogram(size_t label, double value) {

         allPixelsHistogram[label] += value;
     }

    /**
     * traverses the tree to the leaf then increments the instance label's histogram
     */
    const RandomTree<Instance, FeatureFunction>* setAllPixelsHistogram(const Instance& instance) {
           if (isLeaf())
           {
        	   size_t label = instance.getLabel();
        	   allPixelsHistogram[label] += 1;
           return this;
           }

                assert(left.get());
                assert(right.get());

                bool flippedSameSplit;
                if (split.split(instance,flippedSameSplit) == LEFT) {
                    return left->setAllPixelsHistogram(instance);
                } else {
                    return right->setAllPixelsHistogram(instance);
                }
       }

     /**
      * copies the values of allPixelsHistgrams into histograms
      */
     void updateHistograms()
     {

    	 if (isLeaf()) {
     	  for (size_t label = 0; label < numClasses; label++) {
     		// CURFIL_INFO(label<<" histogram[label] "<<histogram[label]<<" allPixelsHistogram[label] "<<allPixelsHistogram[label]);
     		 if (histogram[label] != 0)
     		 { histogram[label] = allPixelsHistogram[label];}
     	   }
    	 return;
    	 }
     	left->updateHistograms();
     	right->updateHistograms();

     }

    /**
     * recomputes the correct histograms, they were changed when checking for flipped features
     */
    void recomputeHistogramNoFlipping(const std::vector<const Instance*>& samples)
    {
        for (size_t label = 0; label < numClasses; label++) {
            histogram[label] = 0;
        }

        for (size_t i = 0; i < samples.size(); i++) {
            histogram[samples[i]->getLabel()] += samples[i]->getWeight();
        }
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
    cuv::ndarray<WeightType, cuv::host_memory_space> allPixelsHistogram;

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

        bool flippedSameSplit;
        if (split.split(instance, flippedSameSplit) == LEFT) {
            return left->traverseToLeaf(instance);
        } else {
            return right->traverseToLeaf(instance);
        }
    }
};

/**
 * A uniform distribution sampler
 * @ingroup rand_sampling
 */
class Sampler {
public:
    
    /**
     * create a Sampler object using a seed and lower and upper bounds
     */
    Sampler(int seed, int lower, int upper) :
            seed(seed), lower(lower), upper(upper), rng(seed), distribution(lower, upper) {
        assert(upper >= lower);
    }

    /**
     * create a Sampler object using another Sampler attributes
     */
    Sampler(const Sampler& other) :
            seed(other.seed), lower(other.lower), upper(other.upper), rng(seed), distribution(lower, upper) {
    }

    /**
     * @return next random value
     */
    int getNext();

    /**
     * @return seed for the random generator
     */
    int getSeed() const {
        return seed;
    }

    /**
     * @return lower bound of the distribution
     */
    int getLower() const {
        return upper;
    }

    /**
     * @return upper bound of the distribution
     */
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

/**
 * Stores samples (images or pixel instances), if max size is reached, a sample replaces another chosen at random
 * @ingroup rand_sampling
 */
template<class T>
class ReservoirSampler {
public:

    ReservoirSampler() :
            count(0), samples(0), reservoir() {
    }

    /**
     * create a ReservoirSampler object with the given reservoir size
     */
    ReservoirSampler(size_t samples) :
            count(0), samples(samples), reservoir() {
        reservoir.reserve(samples);
        assert(reservoir.empty());
    }

    /**
     * add the given sample to the reservoir or replace another if max size was reached
     */
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

    /**
     * @return the reservoir of samples
     */
    const std::vector<T>& getReservoir() const {
        return reservoir;
    }

private:
    size_t count;
    size_t samples;
    std::vector<T> reservoir;
};

/**
 * Used to get a uniform sampler after incrementing the seed
 * @ingroup rand_sampling
 */
// Training-time classes
class RandomSource {

private:
    int seed;

public:
    
    /**
     * create a random  generator with the given seed
     */
    RandomSource(int seed) :
            seed(seed) {
    }

    /**
     * create a uniform sampler with the given upper bound
     */
    Sampler uniformSampler(int val) {
        return uniformSampler(0, val - 1);
    }

    /**
     * create a uniform sampler with the given lower and upper bounds
     */
    Sampler uniformSampler(int lower, int upper) {
        return Sampler(seed++, lower, upper);
    }
};

/**
 * Class that does the actual training of a tree
 * @ingroup training_helper_classes
 */
template<class Instance, class FeatureEvaluation, class FeatureFunction>
class RandomTreeTrain {
public:
    /** featureCount: K >= 1, the number of random splits to sample and evaluate
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

    void compareHistograms(cuv::ndarray<WeightType, cuv::host_memory_space> allHistogram,
    		cuv::ndarray<WeightType, cuv::host_memory_space> leftHistogram,
    		cuv::ndarray<WeightType, cuv::host_memory_space> rightHistogram,
             const SplitFunction<Instance, FeatureFunction>& bestSplit, size_t histogramSize) const {

         const unsigned int leftRightStride = 1; // consecutive in memory

         // assert that actual score is the same as the calculated score
         double totalLeft = sum(leftHistogram);
         double totalRight = sum(rightHistogram);

         WeightType leftHistogramArray[histogramSize];
         WeightType rightHistogramArray[histogramSize];
         WeightType allHistogramArray[histogramSize];

         std::stringstream strLeft;
         std::stringstream strRight;
         std::stringstream strAll;

         for (size_t i=0; i<histogramSize; i++)
         {
        	 leftHistogramArray[i] = leftHistogram[i];
        	 rightHistogramArray[i] = rightHistogram[i];
        	 allHistogramArray[i] = allHistogram[i];

             strLeft<<leftHistogramArray[i]<<",";
             strRight<<rightHistogramArray[i]<<",";
             strAll<<allHistogramArray[i]<<",";
         }
         double actualScore = NormalizedInformationGainScore::calculateScore(histogramSize, leftHistogramArray, rightHistogramArray,
                 leftRightStride,
                 allHistogramArray, totalLeft, totalRight);
         double diff = std::fabs(actualScore - bestSplit.getScore());
        if (diff > 0.02) {
             std::ostringstream o;
             o.precision(10);
             o << "actual score and best split score differ: " << diff << std::endl;
             o << "actual score: " << actualScore << std::endl;
             o << "best split score: " << bestSplit.getScore() << std::endl;
             o << "total left: " << totalLeft << std::endl;
             o << "total right: " << totalRight << std::endl;
             o << "histogram:  " << strAll.str() << std::endl;
             o << "histogram left:  " << strLeft.str() << std::endl;
             o << "histogram right: " << strRight.str() << std::endl;
             throw std::runtime_error(o.str());
        }
     }

public:

    /**
     * Train a single random tree breadth-first
     */
    void train(FeatureEvaluation& featureEvaluation,
            RandomSource& randomSource,
            const std::vector<std::pair<RandomTreePointer, Samples> >& samplesPerNode,
            int idNode, int currentLevel = 1) const {

        // Depth exhausted: leaf node
        if (currentLevel == configuration.getMaxDepth()) {
            return;
        }

        CURFIL_INFO("training level " << currentLevel << ". nodes: " << samplesPerNode.size());

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

            size_t numClasses = currentNode->getHistogram().size();
            cuv::ndarray<WeightType, cuv::host_memory_space> allHistogram(numClasses);
            cuv::ndarray<WeightType, cuv::host_memory_space> leftHistogram(numClasses);
            cuv::ndarray<WeightType, cuv::host_memory_space> rightHistogram(numClasses);

            for (size_t c=0; c<numClasses; c++)
            {
           	 leftHistogram[c] = 0;
           	 rightHistogram[c] = 0;
           	 allHistogram[c] = 0;
            }

            unsigned int totalFlipped = 0;
            unsigned int rightFlipped = 0;
            unsigned int leftFlipped = 0;

            for (size_t sample = 0; sample < samples.size(); sample++) {
                assert(samples[sample] != NULL);
                bool flippedSameSplit;
                Instance* ptr = const_cast<Instance *>(samples[sample]);

                SplitBranch splitResult = bestSplit.split(*ptr, flippedSameSplit);

                bool flipSetting = samples[sample]->getFlipping();

            	if (flipSetting && !flippedSameSplit)
            		ptr->setFlipping(false);

            	allHistogram[samples[sample]->getLabel()] += samples[sample]->getWeight();
                if (splitResult == LEFT) {
                	samplesLeft.push_back(ptr);
                	leftHistogram[samples[sample]->getLabel()] += samples[sample]->getWeight();
					if (flipSetting) {
						totalFlipped += 1;
						allHistogram[samples[sample]->getLabel()] += samples[sample]->getWeight();
						if (flippedSameSplit)
							leftHistogram[samples[sample]->getLabel()] += samples[sample]->getWeight();
						else
							{rightHistogram[samples[sample]->getLabel()] += samples[sample]->getWeight();
							rightFlipped += 1;}
					}
                } else {
                	samplesRight.push_back(ptr);
                	rightHistogram[samples[sample]->getLabel()] += samples[sample]->getWeight();
                	if (flipSetting) {
                		totalFlipped += 1;
                		allHistogram[samples[sample]->getLabel()] += samples[sample]->getWeight();
                		if (flippedSameSplit)
                			rightHistogram[samples[sample]->getLabel()] += samples[sample]->getWeight();
                		else
                			{leftHistogram[samples[sample]->getLabel()] += samples[sample]->getWeight();
                			 leftFlipped += 1;}
                	}
                }
            }

            assert(samplesLeft.size() + samplesRight.size() == samples.size());

            boost::shared_ptr<RandomTree<Instance, FeatureFunction> > leftNode = boost::make_shared<RandomTree<Instance,
                    FeatureFunction> >(++idNode, currentNode->getLevel() + 1, samplesLeft, numClasses, currentNode);

            boost::shared_ptr<RandomTree<Instance, FeatureFunction> > rightNode = boost::make_shared<
                    RandomTree<Instance, FeatureFunction> >(++idNode, currentNode->getLevel() + 1, samplesRight,
                    numClasses, currentNode);

#ifndef NDEBUG
            compareHistograms(allHistogram, leftHistogram, rightHistogram, bestSplit, numClasses);
#endif

            bool errorEmptyChildren = false;
            if (samplesLeft.empty() && leftFlipped == 0)
            {
            	errorEmptyChildren = true;
            }
            if (samplesRight.empty() && rightFlipped == 0)
            {
            	errorEmptyChildren = true;
            }
            if (errorEmptyChildren) {
                CURFIL_ERROR("best split score: " << bestSplit.getScore());
                CURFIL_ERROR("samples: " << samples.size());
                CURFIL_ERROR("threshold: " << bestSplit.getThreshold());
                CURFIL_ERROR("feature: " << bestSplit.getFeature());
                CURFIL_ERROR("histogram: " << currentNode->getHistogram());
                CURFIL_ERROR("samplesLeft: " << samplesLeft.size());
                CURFIL_ERROR("samplesRight: " << samplesRight.size());
                CURFIL_ERROR("leftFlipped "<<leftFlipped<<" rightFlipped "<<rightFlipped<<" totalFlipped "<<totalFlipped)

                compareHistograms(allHistogram, leftHistogram, rightHistogram, bestSplit, numClasses);

                if (samplesLeft.empty()) {
                    throw std::runtime_error("no samples in left node");
                }
                if (samplesRight.empty()) {
                    throw std::runtime_error("no samples in right node");
                }
            }

			if (totalFlipped > 0) {
				currentNode->recomputeHistogramNoFlipping(samples);
			}

			if (!samplesLeft.empty() && !samplesRight.empty()) {
				currentNode->addChildren(bestSplit, leftNode, rightNode);

				if (shouldContinueGrowing(leftNode)) {
					samplesPerNodeNextLevel.push_back(std::make_pair(leftNode, samplesLeft));
					if ((currentLevel + 1) == configuration.getMaxDepth() && totalFlipped > 0)
						{leftNode->recomputeHistogramNoFlipping(samplesLeft);}
				}
				else if (totalFlipped > 0) {
					leftNode->recomputeHistogramNoFlipping(samplesLeft);
				}

				if (shouldContinueGrowing(rightNode)) {
					samplesPerNodeNextLevel.push_back(std::make_pair(rightNode, samplesRight));
					if ((currentLevel + 1) == configuration.getMaxDepth() && totalFlipped > 0)
						{rightNode->recomputeHistogramNoFlipping(samplesRight);}
				}
				else if (totalFlipped > 0) {
					rightNode->recomputeHistogramNoFlipping(samplesRight);
				}
			}
			else
				idNode = idNode - 2;
        }




        CURFIL_INFO("training level " << currentLevel << " took " << trainTimer.format(3));
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
