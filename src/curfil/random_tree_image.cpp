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
#include "random_tree_image.h"

#include <boost/format.hpp>
#include <map>
#include <math.h>
#include <set>
#include <tbb/mutex.h>
#include <tbb/parallel_for_each.h>
#include <thrust/gather.h>
#include <thrust/sort.h>

#include "ndarray_ops.h"
#include "random_tree_image_gpu.h"
#include "random_tree.h"
#include "score.h"

namespace curfil {

// for statistics
extern ImageCache imageCache;

class FeatureEvaluationCPU {

public:

    FeatureEvaluationCPU(size_t numClasses,
            size_t numFeatures,
            size_t numThresholds,
            const std::vector<const PixelInstance*>& samples,
            const ImageFeaturesAndThresholds<cuv::host_memory_space>& features,
            tbb::concurrent_vector<cuv::ndarray<WeightType, cuv::host_memory_space>>& perClassHistograms) :
            numClasses(numClasses),
                    numFeatures(numFeatures),
                    numThresholds(numThresholds),
                    samples(samples), features(features), perClassHistograms(perClassHistograms) {

        assert(!samples.empty());
    }

    // must be a const-method for TBB
    void operator()(const tbb::blocked_range<size_t>& range) const {

        cuv::ndarray<WeightType, cuv::host_memory_space> perClassHistogram(
                cuv::extents[numClasses][numFeatures][numThresholds][2]);

        for (size_t i = 0; i < perClassHistogram.size(); i++) {
            perClassHistogram[i] = 0;
        }

        std::vector<ImageFeatureFunction> featureFunctions(numFeatures);
        for (size_t featureNr = 0; featureNr < numFeatures; ++featureNr) {
            featureFunctions[featureNr] = features.getFeatureFunction(featureNr);
        }

        // transpose the thresholds
        std::vector<std::vector<float> > thresholds;
        for (size_t featureNr = 0; featureNr < numFeatures; ++featureNr) {
            std::vector<float> thresholdsPerFeature;
            for (size_t threshNr = 0; threshNr < numThresholds; ++threshNr) {
                thresholdsPerFeature.push_back(features.getThreshold(threshNr, featureNr));
            }
            thresholds.push_back(thresholdsPerFeature);
        }

        assert(perClassHistogram.ndim() == 4);
        const unsigned int labelStride = perClassHistogram.stride(0);
        const unsigned int featureStride = perClassHistogram.stride(1);
        const unsigned int thresholdsStride = perClassHistogram.stride(2);

        for (size_t s = range.begin(); s != range.end(); s++) {
            const PixelInstance* sample = samples[s];
            assert(sample);
            const LabelType label = sample->getLabel();
            const WeightType weight = sample->getWeight();

            const unsigned int labelOffset = label * labelStride;

            HorizontalFlipSetting horFlipSetting = sample->getHorFlipSetting();
            double value1;
            double value2 = 0;

            for (size_t featureNr = 0; featureNr < numFeatures; ++featureNr) {
            	switch (horFlipSetting)
            	{
            	case NoFlip:
            		value1 = featureFunctions[featureNr].calculateFeatureResponse(*sample, false);
            		break;
            	case Flip:
            	    value1 = featureFunctions[featureNr].calculateFeatureResponse(*sample, true);
            	    break;
            	case Both:
            		value1 = featureFunctions[featureNr].calculateFeatureResponse(*sample, false);
            		value2 = featureFunctions[featureNr].calculateFeatureResponse(*sample, true);
            		break;
            	default:
            		value1 = featureFunctions[featureNr].calculateFeatureResponse(*sample, false);
            	}

                const std::vector<float>& thresholdsPerFeature = thresholds[featureNr];

                const unsigned int featureOffset = labelOffset + featureNr * featureStride;

                for (size_t threshNr = 0; threshNr < numThresholds; ++threshNr) {
                    const float threshold = thresholdsPerFeature[threshNr];
                    assert(!isnan(threshold));

                    // avoid the if () else () branch here by casting the compare into 0 or 1
                    // important: (!(x<=y)) is not the same as (x>y) because of NaNs!
                    int offset = static_cast<int>(!(value1 <= threshold));
                    assert(offset == ((value1 <= threshold) ? 0 : 1));

                    // offset stride must be 1
                    assert(perClassHistogram.stride(3) == 1);
                    unsigned int idx = featureOffset + threshNr * thresholdsStride + offset;
                    perClassHistogram.ptr()[idx] += weight;
                    assert(perClassHistogram(label, featureNr, threshNr, offset) == perClassHistogram.ptr()[idx]);

					if (horFlipSetting == Both) {
						offset = static_cast<int>(!(value2 <= threshold));
						assert(offset == ((value2 <= threshold) ? 0 : 1));
						idx = featureOffset + threshNr * thresholdsStride + offset;
						perClassHistogram.ptr()[idx] += weight;
						assert(perClassHistogram(label, featureNr, threshNr,offset)== perClassHistogram.ptr()[idx]);
					}
                }
            }
        }

        perClassHistograms.push_back(perClassHistogram);
    }

private:
    const size_t numClasses;
    const size_t numFeatures;
    const size_t numThresholds;
    const std::vector<const PixelInstance*>& samples;
    const ImageFeaturesAndThresholds<cuv::host_memory_space>& features;
    tbb::concurrent_vector<cuv::ndarray<WeightType, cuv::host_memory_space> >& perClassHistograms;
};

bool ImageFeatureFunction::operator==(const ImageFeatureFunction& other) const {
    if (featureType != other.featureType)
        return false;
    if (offset1 != other.offset1)
        return false;
    if (region1 != other.region1)
        return false;
    if (channel1 != other.channel1)
        return false;
    if (offset2 != other.offset2)
        return false;
    if (region2 != other.region2)
        return false;
    if (channel2 != other.channel2)
        return false;

    return true;

}

template<>
cuv::ndarray<ScoreType, cuv::host_memory_space> ImageFeatureEvaluation::calculateScores(
        const cuv::ndarray<WeightType, cuv::host_memory_space>& counters,
        const ImageFeaturesAndThresholds<cuv::host_memory_space>& featuresAndThresholds,
        const cuv::ndarray<WeightType, cuv::host_memory_space>& histogram) {

    const unsigned int numFeatures = configuration.getFeatureCount();
    const unsigned int numThresholds = configuration.getThresholds();

    cuv::ndarray<ScoreType, cuv::host_memory_space> scores(numThresholds, numFeatures, scoresAllocator);

    utils::Profile profile("calculateScores");

    const size_t numLabels = histogram.size();
    assert(numLabels > 0);

    for (size_t featureNr = 0; featureNr < numFeatures; featureNr++) {

        for (size_t threshNr = 0; threshNr < numThresholds; ++threshNr) {

            assert(!isnan(featuresAndThresholds.getThreshold(threshNr, featureNr)));

            ScoreType summedLeft = 0;
            ScoreType summedRight = 0;

            WeightType perClassLeftCounter[numLabels];
            WeightType perClassRightCounter[numLabels];
            const unsigned int leftRightStride = 1; // consecutive in memory

            for (size_t classNr = 0; classNr < numLabels; classNr++) {

                const WeightType& left = counters(classNr, featureNr, threshNr, 0);
                const WeightType& right = counters(classNr, featureNr, threshNr, 1);

                perClassLeftCounter[classNr] = left;
                perClassRightCounter[classNr] = right;

                summedLeft += left;
                summedRight += right;
            }

            ScoreType score = NormalizedInformationGainScore::calculateScore(numLabels, perClassLeftCounter,
                    perClassRightCounter, leftRightStride, histogram.ptr(), summedLeft, summedRight);

            scores(threshNr, featureNr) = score;
        }
    }

    return scores;
}

std::vector<SplitFunction<PixelInstance, ImageFeatureFunction> > ImageFeatureEvaluation::evaluateBestSplits(
        RandomSource& randomSource,
        const std::vector<std::pair<boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> >,
                std::vector<const PixelInstance*> > >& samplesPerNode) {

    std::vector<SplitFunction<PixelInstance, ImageFeatureFunction> > bestSplits(samplesPerNode.size());

    ImageFeaturesAndThresholds<cuv::host_memory_space> featuresAndThresholdsCPU(configuration.getFeatureCount(),
            configuration.getThresholds(), featuresAllocator);

    ImageFeaturesAndThresholds<cuv::dev_memory_space> featuresAndThresholdsGPU(configuration.getFeatureCount(),
            configuration.getThresholds(), featuresAllocator);

    size_t totalTransferTimeMicrosecondsStart = imageCache.getTotalTransferTimeMircoseconds();

    const AccelerationMode accelerationMode = configuration.getAccelerationMode();

    utils::Timer generatingRandomFeaturesTimer;

    {
        std::set<const RGBDImage*> images;
        std::vector<const PixelInstance*> allSamples;
        unsigned int numSkipped = 0;
        for (size_t nodeId = 0; nodeId < samplesPerNode.size(); nodeId++) {

            const std::vector<const PixelInstance*>& samples = samplesPerNode[nodeId].second;
            for (size_t s = 0; s < samples.size(); s++) {
                const PixelInstance* sample = samples[s];
                if (accelerationMode == GPU_ONLY || accelerationMode == GPU_AND_CPU_COMPARE) {
                    if (images.size() == static_cast<size_t>(configuration.getImageCacheSize())) {
                        if (images.find(sample->getRGBDImage()) == images.end()) {
                            // skip sample. image does not fit in cache
                            numSkipped++;
                            continue;
                        }
                    }
                }
                images.insert(sample->getRGBDImage());
                allSamples.push_back(sample);
            }

        }

        if (numSkipped > 0) {
            CURFIL_INFO("randomFeatureGeneration: skipped " << numSkipped << " samples");
        }

        const int seed = randomSource.uniformSampler(0xFFFFFF).getNext();

        if (accelerationMode == CPU_ONLY || accelerationMode == GPU_AND_CPU_COMPARE) {
            featuresAndThresholdsCPU = generateRandomFeatures(allSamples, seed, true, cuv::host_memory_space());
        }
        if (accelerationMode == GPU_ONLY || accelerationMode == GPU_AND_CPU_COMPARE) {
            featuresAndThresholdsGPU = generateRandomFeatures(allSamples, seed, true, cuv::dev_memory_space());
        }
    }

    CURFIL_INFO("generating random features: " << generatingRandomFeaturesTimer.format(2));

    tbb::mutex cpuEvaluationMutex;

    size_t grainSize = 1;
    if (accelerationMode != CPU_ONLY) {
        // GPU: use two threads
        grainSize = ceil(samplesPerNode.size() / 2.0);
    }
    tbb::parallel_for(tbb::blocked_range<size_t>(0, samplesPerNode.size(), grainSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for(size_t nodeNr = range.begin(); nodeNr != range.end(); nodeNr++) {

                    const std::pair<boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> >,
                    std::vector<const PixelInstance*> >& nodeSamples = samplesPerNode[nodeNr];

                    utils::Timer timerEvaluateBestSplit;

                    RandomTree<PixelInstance, ImageFeatureFunction>& currentNode = *(nodeSamples.first);
                    const std::vector<const PixelInstance*>& samples = nodeSamples.second;

                    const size_t numLabels = currentNode.getNumClasses();
                    assert(numLabels >= 2 && numLabels < 256);
                    assert(!samples.empty());

                    SplitFunction<PixelInstance, ImageFeatureFunction> bestFeatureCPU;
                    SplitFunction<PixelInstance, ImageFeatureFunction> bestFeatureGPU;

                    cuv::ndarray<ScoreType, cuv::host_memory_space> scoresCPU;
                    cuv::ndarray<ScoreType, cuv::host_memory_space> scoresGPU;

                    size_t transferTimeStart = imageCache.getTotalTransferTimeMircoseconds();

                    if (accelerationMode == CPU_ONLY || accelerationMode == GPU_AND_CPU_COMPARE) {

                        CURFIL_DEBUG("prepare batches");

                        std::vector<std::vector<const PixelInstance*> > batches = prepare(samples, currentNode,
                                cuv::host_memory_space());

                        tbb::concurrent_vector<cuv::ndarray<WeightType, cuv::host_memory_space> > perClassHistograms;
                        utils::Timer timeEvaluate;

                        int numThreads = configuration.getNumThreads();

                        assert(numThreads > 0);

                        size_t grainSize = std::max(100ul, samples.size() / numThreads);

                        CURFIL_DEBUG("start evaluation");

                        cuv::ndarray<WeightType, cuv::host_memory_space> countersCPU(
                                cuv::extents[numLabels][configuration.getFeatureCount()][configuration.getThresholds()][2]);
                        for (size_t i = 0; i < countersCPU.size(); i++) {
                            countersCPU[i] = 0;
                        }

                        {
                            // tbb::mutex::scoped_lock cpuEvaluationLock(cpuEvaluationMutex);

                            utils::Profile profile("feature evaluation CPU");

                            FeatureEvaluationCPU evaluation(numLabels, configuration.getFeatureCount(),
                                    configuration.getThresholds(),
                                    samples, featuresAndThresholdsCPU, perClassHistograms);

                            tbb::parallel_for(tbb::blocked_range<size_t>(0, samples.size(), grainSize), evaluation);
                            currentNode.setTimerValue("featureEvaluation", profile.getSeconds());

                            assert(!perClassHistograms.empty());

                            utils::Profile reaggregateHistograms("reaggregateHistograms");
                            for (size_t i = 0; i < perClassHistograms.size(); i++) {
                                countersCPU += perClassHistograms[i];
                            }
                            currentNode.setTimerValue("reaggregateHistograms", reaggregateHistograms.getSeconds());
                        }

                        CURFIL_DEBUG("calculate scores");

                        {
                            utils::Profile profile("calculateScores");
                            scoresCPU = calculateScores(countersCPU, featuresAndThresholdsCPU, currentNode.getHistogram());
                            currentNode.setTimerValue("calculateScores", profile.getSeconds());
                        }

                        currentNode.setTimerValue("evaluationTime", timeEvaluate);
                    }
                    cuv::ndarray<WeightType, cuv::dev_memory_space> counters2;
                    if (accelerationMode == GPU_ONLY || accelerationMode == GPU_AND_CPU_COMPARE) {

                        std::vector<std::vector<const PixelInstance*> > batches = prepare(samples, currentNode,
                                cuv::dev_memory_space());

                        if (accelerationMode == GPU_AND_CPU_COMPARE) {
                            featuresAndThresholdsGPU = featuresAndThresholdsCPU;
                        }

                        utils::Timer featureResponsesAndHistograms;

                        cuv::ndarray<WeightType, cuv::dev_memory_space> counters = calculateFeatureResponsesAndHistograms(
                                currentNode, batches, featuresAndThresholdsGPU);

                        currentNode.setTimerValue("featureResponsesAndHistograms", featureResponsesAndHistograms);

                        utils::Timer calculateScoresTimer;

                        cuv::ndarray<WeightType, cuv::dev_memory_space> histogram = currentNode.getHistogram();
                        scoresGPU = calculateScores(counters, featuresAndThresholdsGPU, histogram);

                        currentNode.setTimerValue("calculateScores", calculateScoresTimer);

                        counters2 = counters;
                    }

                    size_t transferTimeEnd = imageCache.getTotalTransferTimeMircoseconds();

                    currentNode.setTimerValue("transferImages", (transferTimeEnd - transferTimeStart) / static_cast<double>(1e6));

                    currentNode.setTimerValue("evaluateBestSplit", timerEvaluateBestSplit);

                    if (accelerationMode == GPU_AND_CPU_COMPARE) {
                        if (scoresCPU.shape() != scoresGPU.shape()) {
                            throw std::runtime_error("different shapes");
                        }
                        for (size_t i = 0; i < scoresCPU.size(); i++) {
                            ScoreType diff = abs(static_cast<ScoreType>(scoresCPU[i] - scoresGPU[i]));
                            if (diff > 1e-10) {
                                const std::string message = boost::str(
                                        boost::format("different scores [%d]: %.10f vs. %.10f, diff: %.10f")
                                        % i
                                        % scoresCPU[i]
                                        % scoresGPU[i]
                                        % diff);
                                throw std::runtime_error(message);
                            }
                        }
                    }

                    cuv::ndarray<ScoreType, cuv::host_memory_space> scores;
                    if (accelerationMode == CPU_ONLY || accelerationMode == GPU_AND_CPU_COMPARE) {
                        scores = scoresCPU;
                    } else {
                        scores = scoresGPU;
                    }

                    assert(scores.ndim() == 2);
                    assert(scores.shape(0) == configuration.getThresholds());
                    assert(scores.shape(1) == configuration.getFeatureCount());
                    ScoreType bestScore = -std::numeric_limits<ScoreType>::infinity();
                    uint16_t bestThresh = 0;
                    unsigned int bestFeat = 0;
                    for (uint16_t thresh = 0; thresh < configuration.getThresholds(); thresh++) {
                        for (unsigned int feat = 0; feat < configuration.getFeatureCount(); feat++) {
                            const ScoreType score = scores(thresh, feat);
                            if (isnan(bestScore) || detail::isScoreBetter(bestScore, score, feat)) {
                                bestFeat = feat;
                                bestThresh = thresh;
                                bestScore = score;
                            }
                        }
                    }

                    assert(bestScore > 0.0);

                    ImageFeatureFunction feature;
                    float threshold;
                    if (accelerationMode == CPU_ONLY) {
                        feature = featuresAndThresholdsCPU.getFeatureFunction(bestFeat);
                        threshold = featuresAndThresholdsCPU.getThreshold(bestThresh, bestFeat);
                    } else {
                        feature = featuresAndThresholdsGPU.getFeatureFunction(bestFeat);
                        threshold = featuresAndThresholdsGPU.getThreshold(bestThresh, bestFeat);
                    }

                    SplitFunction<PixelInstance, ImageFeatureFunction> bestFeature(bestFeat, feature, threshold, bestScore);

                    CURFIL_DEBUG("tree " << currentNode.getTreeId() << ", node " << currentNode.getNodeId() <<
                            ", best score: " << bestScore << ", " << feature);

                   unsigned int index = bestFeat * configuration.getThresholds() * currentNode.getNumClasses() * 2;
                   index += bestThresh * currentNode.getNumClasses() * 2;

               /*   std::stringstream strLeft;
                    std::stringstream strRight;
                    for (size_t label = 0; label < currentNode.getNumClasses(); label++) {
                    	strLeft<<counters2[index + label * 2 + 0]<<",";
                    	strRight<<counters2[index + label * 2 + 1]<<",";
                    }
                    CURFIL_INFO("org histogram"<<currentNode.getHistogram());
                    CURFIL_INFO("left split "<<strLeft.str());
                    CURFIL_INFO("right split"<<strRight.str());*/
                    bestSplits[nodeNr]= bestFeature;
                }
            });

    size_t totalTransferTimeMicrosecondsEnd = imageCache.getTotalTransferTimeMircoseconds();
    assert(totalTransferTimeMicrosecondsEnd >= totalTransferTimeMicrosecondsStart);
    double transferTime = (totalTransferTimeMicrosecondsEnd - totalTransferTimeMicrosecondsStart)
            / static_cast<double>(1e6);

    if (transferTime > 0) {
        CURFIL_INFO((boost::format("image cache transfer time: %.3f s") % transferTime).str());
    }

    return bestSplits;
}

template<>
void ImageFeatureEvaluation::sortFeatures(
        ImageFeaturesAndThresholds<cuv::host_memory_space>& featuresAndThresholds,
        const cuv::ndarray<int, cuv::host_memory_space>& keysIndices) const {

    const size_t numFeatures = configuration.getFeatureCount();
    assert(featuresAndThresholds.m_features.shape(1) == numFeatures);

    ImageFeaturesAndThresholds<cuv::host_memory_space> tmpFeaturesAndThresholds = featuresAndThresholds.copy();

    int* keys = keysIndices[cuv::indices[0][cuv::index_range()]].ptr();
    int* indices = keysIndices[cuv::indices[1][cuv::index_range()]].ptr();

    thrust::sort_by_key(keys, keys + numFeatures, indices);

    cuv::ndarray<int8_t, cuv::host_memory_space> features = tmpFeaturesAndThresholds.features();
    cuv::ndarray<int8_t, cuv::host_memory_space> sortedFeatures = featuresAndThresholds.features();

    assert(features.shape() == sortedFeatures.shape());

    const size_t dim = features.shape(0);
    assert(dim == 11);
    for (size_t d = 0; d < dim; d++) {
        int8_t* ptr = features[cuv::indices[d][cuv::index_range()]].ptr();
        int8_t* sortedPtr = sortedFeatures[cuv::indices[d][cuv::index_range()]].ptr();

        assert(ptr != sortedPtr);
        thrust::gather(indices, indices + numFeatures, ptr, sortedPtr);
    }

#ifndef NDEBUG
    for (size_t feat = 0; feat < numFeatures; feat++) {
        const ImageFeatureFunction feature1 = tmpFeaturesAndThresholds.getFeatureFunction(indices[feat]);
        const ImageFeatureFunction feature2 = featuresAndThresholds.getFeatureFunction(feat);
        assert(feature1 == feature2);
    }
#endif

#ifndef NDEBUG
    for (size_t feat = 1; feat < numFeatures; feat++) {
        const ImageFeatureFunction feature1 = featuresAndThresholds.getFeatureFunction(feat - 1);
        const ImageFeatureFunction feature2 = featuresAndThresholds.getFeatureFunction(feat);

        const int key1 = feature1.getSortKey();
        const int key2 = feature2.getSortKey();
        assert(key1 <= key2);
    }
#endif

    assert(featuresAndThresholds.thresholds().ndim() == 2);
    assert(featuresAndThresholds.thresholds().shape(0) == configuration.getThresholds());
    assert(featuresAndThresholds.thresholds().shape(1) == numFeatures);

    for (size_t thresh = 0; thresh < configuration.getThresholds(); thresh++) {
        float* thresholdsPtr =
                tmpFeaturesAndThresholds.m_thresholds[cuv::indices[thresh][cuv::index_range()]].ptr();
        float* sortedThresholdsPtr =
                featuresAndThresholds.m_thresholds[cuv::indices[thresh][cuv::index_range()]].ptr();

        thrust::gather(indices, indices + numFeatures, thresholdsPtr, sortedThresholdsPtr);
    }

}

std::vector<std::vector<const PixelInstance*> > ImageFeatureEvaluation::prepare(
        const std::vector<const PixelInstance*>& samples,
        RandomTree<PixelInstance, ImageFeatureFunction>& node,
        cuv::host_memory_space) {
    std::vector<std::vector<const PixelInstance*> > batches;
    batches.push_back(samples);
    return batches;
}

ImageFeaturesAndThresholds<cuv::host_memory_space> ImageFeatureEvaluation::generateRandomFeatures(
        const std::vector<const PixelInstance*>& samples, int seed, const bool sort, cuv::host_memory_space) {

    utils::Profile profile("generateRandomFeatures host");

    unsigned int numFeatures = configuration.getFeatureCount();
    unsigned int numThresholds = configuration.getThresholds();

    ImageFeaturesAndThresholds<cuv::host_memory_space> featuresAndThresholds(numFeatures, numThresholds,
            featuresAllocator);

    assert(!samples.empty());

    RandomSource randomSource(seed);

    Sampler sampleGen = randomSource.uniformSampler(samples.size());

    size_t generatedFeatures = 0;

    cuv::ndarray<int, cuv::host_memory_space> keysIndices(2, numFeatures, keysIndicesAllocator);

    int maxLoops = 3 * numFeatures;

    while (generatedFeatures < numFeatures) {
        const ImageFeatureFunction feature = sampleFeature(randomSource, samples);

        if (--maxLoops <= 0) {
            throw std::runtime_error("failed to generate random features. max loops exceeded");
        }

        if (!feature.isValid()) {
            continue;
        }

        bool isValid = true;
        for (size_t thresh = 0; thresh < configuration.getThresholds(); thresh++) {
            size_t maxTries = 10;
            float threshold;
            do {
                const PixelInstance* sample = samples.at(sampleGen.getNext());
                assert(sample);
                threshold = feature.calculateFeatureResponse(*sample);
            } while (isnan(threshold) && --maxTries > 0);

            if (isnan(threshold)) {
                isValid = false;
                break;
            }

            featuresAndThresholds.thresholds()(thresh, generatedFeatures) = threshold;
            assert(featuresAndThresholds.getThreshold(thresh, generatedFeatures) == threshold);
        }

        if (!isValid) {
            continue;
        }

        featuresAndThresholds.setFeatureFunction(generatedFeatures, feature);

        keysIndices(0, generatedFeatures) = feature.getSortKey();
        keysIndices(1, generatedFeatures) = generatedFeatures;

        generatedFeatures++;
    }

    if (sort)
        sortFeatures(featuresAndThresholds, keysIndices);

    return featuresAndThresholds;
}

const ImageFeatureFunction ImageFeatureEvaluation::sampleFeature(RandomSource& randomSource,
        const std::vector<const PixelInstance*>&) const {
    // Random feature type
    Sampler rgen_ft = randomSource.uniformSampler(2);

    uint16_t boxRadius = configuration.getBoxRadius();
    uint16_t regionSize = configuration.getRegionSize();
    bool useDepthImages = configuration.isUseDepthImages();

    // (x,y) offsets
    Sampler rgen_xy = randomSource.uniformSampler(-boxRadius, boxRadius);

    // (w,h) regions
    Sampler rgen_region = randomSource.uniformSampler(1, regionSize);

    // Color channel [0-2]
    Sampler rgen_cc = randomSource.uniformSampler(3);

    // Depth/height channel (0/2)
    Sampler rgen_dh = randomSource.uniformSampler(2);

    // Generate a feature that is non-constant
    Offset offset1(rgen_xy.getNext(), rgen_xy.getNext());

    // Second, disjoint pixel
    Offset offset2;
    do {
        offset2 = Offset(rgen_xy.getNext(), rgen_xy.getNext());
    } while (offset1 == offset2);

    Region region1(rgen_region.getNext(), rgen_region.getNext());
    Region region2(rgen_region.getNext(), rgen_region.getNext());

    assert(region1.getX() >= 0);
    assert(region1.getY() >= 0);
    assert(region2.getX() >= 0);
    assert(region2.getY() >= 0);

    int depth_off = 0;

	if (useDepthImages) {
		switch (rgen_ft.getNext()) {
		case 0:
          depth_off = rgen_dh.getNext() ? 2 : 0; // compare only within depth or within height
			return ImageFeatureFunction(DEPTH, offset1, region1, depth_off, offset2,
					region2, depth_off);
		case 1:
			return ImageFeatureFunction(COLOR, offset1, region1,
					rgen_cc.getNext(), offset2, region2, rgen_cc.getNext());
		default:
			assert(false);
			break;
		}
	} else
		return ImageFeatureFunction(COLOR, offset1, region1, rgen_cc.getNext(),
				offset2, region2, rgen_cc.getNext());

    throw std::runtime_error("");
}

RandomTreeImage::RandomTreeImage(int id, const TrainingConfiguration& configuration) :
        finishedTraining(false), id(id), configuration(configuration),
                tree(), classLabelPriorDistribution() {
}

RandomTreeImage::RandomTreeImage(boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> > tree,
        const TrainingConfiguration& configuration,
        const cuv::ndarray<WeightType, cuv::host_memory_space>& classLabelPriorDistribution) :
        finishedTraining(true), id(tree->getNodeId()), configuration(configuration),
                tree(tree), classLabelPriorDistribution(classLabelPriorDistribution) {
}

void RandomTreeImage::doTrain(RandomSource& randomSource, size_t numClasses,
        std::vector<const PixelInstance*>& subsamples) {

    RandomTreeTrain<PixelInstance, ImageFeatureEvaluation, ImageFeatureFunction> treeTrain(getId(), numClasses,
            configuration);

    tree = boost::make_shared<RandomTree<PixelInstance, ImageFeatureFunction> >(getId(), 1, subsamples, numClasses); // no parent
    assert(tree->isRoot());

    std::vector<
            std::pair<boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> >,
                    std::vector<const PixelInstance*> > > samplesPerNode;
    samplesPerNode.push_back(std::make_pair(tree, subsamples));

    ImageFeatureEvaluation featureEvaluation(tree->getTreeId(), configuration);
    treeTrain.train(featureEvaluation, randomSource, samplesPerNode, getId());
}

void RandomTreeImage::normalizeHistograms(const double histogramBias, const bool useLabelsPrior) {
    tree->normalizeHistograms(classLabelPriorDistribution, histogramBias, useLabelsPrior);
}

bool RandomTreeImage::shouldIgnoreLabel(const LabelType& label) const {
    const RGBColor color = LabelImage::decodeLabel(label);
    for (const std::string colorString : configuration.getIgnoredColors()){  
    if (color == RGBColor(colorString)) {
            return true;
        }
    }
    return false;
}

void RandomTreeImage::train(const std::vector<LabeledRGBDImage>& trainLabelImages,
        RandomSource& randomSource, size_t subsampleCount, size_t numLabels) {

    assert(subsampleCount > 0);
    assert(finishedTraining == false);
    assert(tree == NULL);
    assert(!trainLabelImages.empty());

    classLabelPriorDistribution.resize(numLabels);
    calculateLabelPriorDistribution(trainLabelImages);

    // Subsample training set
    std::vector<PixelInstance> subsamples;

    if (configuration.getSubsamplingType() == "pixelUniform") {
        subsamples = subsampleTrainingDataPixelUniform(trainLabelImages, randomSource, subsampleCount);
    } else if (configuration.getSubsamplingType() == "classUniform") {
        subsamples = subsampleTrainingDataClassUniform(trainLabelImages, randomSource, subsampleCount);
    } else {
        throw std::runtime_error(
                boost::str(boost::format("unknown subsamplingType: %d") % configuration.getSubsamplingType()));
    }

    CURFIL_INFO("sorting " << subsamples.size() << " samples");

    utils::Timer sortTimer;

    // interesting fact: sorting the concrete instances is faster than sorting with pointers!
    // sort by imageid, class id (to improve CPU caching)

    const int SORT_BLOCK_SIZE = 200000;
    auto iterator = subsamples.begin();
    auto end = std::min(subsamples.end(), iterator + SORT_BLOCK_SIZE);
    while (iterator < end) {
        std::sort(iterator, end, [](const PixelInstance& a, const PixelInstance& b) {
            if (a.getRGBDImage() != b.getRGBDImage()) {
                return(a.getRGBDImage() < b.getRGBDImage());
            }

            if (a.getLabel() != b.getLabel()) {
                return (a.getLabel() < b.getLabel());
            }

            // FIXME optimize magic value
            const int quantisation = 10;
            if (a.getY()/quantisation != b.getY()/quantisation) {
                return (a.getY()/quantisation < b.getY()/quantisation);
            }
            if (a.getX()/quantisation != b.getX()/quantisation) {
                return (a.getX()/quantisation < b.getX()/quantisation);
            }
            return false;
        });
        iterator += SORT_BLOCK_SIZE;
        end += SORT_BLOCK_SIZE;
        end = std::min(subsamples.end(), end);
    }

    sortTimer.stop();

    CURFIL_INFO("sorted in " << sortTimer.format(2));

    std::vector<const PixelInstance*> subsamplePointers;
    subsamplePointers.reserve(subsamples.size());
    for (const PixelInstance& sample : subsamples) {
        subsamplePointers.push_back(&sample);
    }

    utils::Timer trainTimer;

    assert(tree == NULL);

    const size_t numClasses = classLabelPriorDistribution.size();
    doTrain(randomSource, numClasses, subsamplePointers);
    assert(tree != NULL);
    finishedTraining = true;

    CURFIL_INFO("trained tree " << getId() << " in " << trainTimer.format(2));
}

void RandomTreeImage::test(const RGBDImage* image, LabelImage& prediction) const {
    assert(finishedTraining);
    assert(image->getWidth() == prediction.getWidth());
    assert(image->getHeight() == prediction.getHeight());

    // Classify one pixel at a time (this may be inefficient)
    for (int y = 0; y < image->getHeight(); ++y) {
        for (int x = 0; x < image->getWidth(); ++x) {
            PixelInstance pixel(image, 0, x, y);
            prediction.setLabel(x, y, tree->classify(pixel));
        }
    }
}

std::vector<PixelInstance> RandomTreeImage::subsampleTrainingDataPixelUniform(
        const std::vector<LabeledRGBDImage>& trainLabelImages,
        RandomSource& randomSource,
        size_t subsampleCount) const {
    // Random across imags type [0..(n-1)]
    auto rgen_image = randomSource.uniformSampler(trainLabelImages.size());

    HorizontalFlipSetting horFlipSetting;
    if (configuration.doHorizontalFlipping())
    	horFlipSetting = Both;
    else
    	horFlipSetting = NoFlip;

    std::vector<PixelInstance> subsamples;
    for (size_t n = 0; n < subsampleCount * trainLabelImages.size(); ++n) {
        unsigned int image_id = rgen_image.getNext();
        assert(image_id < trainLabelImages.size());

        // Sample a random (x,y) coordinate for the image
        const auto image = trainLabelImages[image_id].getLabelImage();

        auto rgen_x = randomSource.uniformSampler(image.getWidth());
        auto rgen_y = randomSource.uniformSampler(image.getHeight());

        do {
            uint16_t x = static_cast<uint16_t>(rgen_x.getNext());
            uint16_t y = static_cast<uint16_t>(rgen_y.getNext());
            assert(x < image.getWidth());
            assert(y < image.getHeight());

            LabelType label = image.getLabel(x, y);
      
            if (shouldIgnoreLabel(label)) {
                continue;
            }

            // Append to sample list
            subsamples.push_back(PixelInstance(&(trainLabelImages[image_id].getRGBDImage()), label, x, y, horFlipSetting));
            break;
        } while (true);
    }

    return subsamples;
}

void RandomTreeImage::calculateLabelPriorDistribution(const std::vector<LabeledRGBDImage>& trainLabelImages) {

    std::map<LabelType, size_t> priorDistribution;
    for (const auto& trainLabelImage : trainLabelImages) {

        const auto& img = trainLabelImage.getLabelImage();

        for (int y = 0; y < img.getHeight(); ++y) {
            for (int x = 0; x < img.getWidth(); ++x) {
                LabelType label = img.getLabel(x, y);
                assert(label >= 0);
                priorDistribution[label]++;
            }
        }
    }

	if (classLabelPriorDistribution.size() < priorDistribution.size()) {
		classLabelPriorDistribution.resize(priorDistribution.size());
	}

	for (LabelType label = 0; label < priorDistribution.size(); label++) {
		classLabelPriorDistribution[label] = priorDistribution[label];
    }

}

std::vector<PixelInstance> RandomTreeImage::subsampleTrainingDataClassUniform(
        const std::vector<LabeledRGBDImage>& trainLabelImages,
        RandomSource& randomSource,
        size_t subsampleCount) const {

    if (trainLabelImages.empty()) {
        throw std::runtime_error("got no label images");
    }

    utils::Timer samplingTimer;

    size_t numLabels = classLabelPriorDistribution.size();

    std::map<RGBColor, LabelType> colorLabelMap;
    for (LabelType label = 0; label < numLabels; label++) {
        colorLabelMap[LabelImage::decodeLabel(label)] = label;
    }

    for (const auto& color : configuration.getIgnoredColors()) {
        if (colorLabelMap.find(color) == colorLabelMap.end()) {
            throw std::runtime_error((boost::format("color to ignore was not found: RGB(%s) (numLabels: %d)")
                    % color % classLabelPriorDistribution.size()).str());
        }
        assert(numLabels >= 1);
        numLabels--;
    }

    // Number of samples per class, rounded up
    const size_t samplesPerClass = static_cast<size_t>(
            ceil(trainLabelImages.size() * static_cast<double>(subsampleCount) /
                    static_cast<double>(numLabels)));

    std::vector<PixelInstance> allSubsamples;

    CURFIL_INFO("sampling " << numLabels << " classes. " << samplesPerClass << " samples per class with "
            << configuration.getNumThreads() << " threads from " << trainLabelImages.size() << " images");

    tbb::concurrent_vector<std::map<LabelType, ReservoirSampler<PixelInstance> > > samplersPerLabel;

    int grainSize = ceil(trainLabelImages.size() / static_cast<double>(configuration.getNumThreads()));

    const int randomSeed = randomSource.uniformSampler(0xFFFFFF).getNext();

    std::set<LabelType> labelsToIgnore;
    for (const std::string colorString : configuration.getIgnoredColors()) {
        LabelType label = std::numeric_limits<LabelType>::max();
        label = getOrAddColorId(RGBColor(colorString), label);
        if (label == std::numeric_limits<LabelType>::max()) {
            throw std::runtime_error(std::string("unknown color to ignore: ") + colorString);
        }
        labelsToIgnore.insert(label);
    }

    HorizontalFlipSetting horFlipSetting;
    if (configuration.doHorizontalFlipping())
    	horFlipSetting = Both;
    else
        horFlipSetting = NoFlip;

    // Parallel Reservoir Sampling
    tbb::parallel_for(tbb::blocked_range<size_t>(0, trainLabelImages.size(), grainSize),
            [&](const tbb::blocked_range<size_t>& range) {

#ifndef NDEBUG
            size_t numImages = range.end() - range.begin();
            assert(numImages > 0);
            assert(numImages <= trainLabelImages.size());
#endif

            std::map<LabelType, ReservoirSampler<PixelInstance> > reservoirSamplers;

            for (LabelType label = 0; label < classLabelPriorDistribution.size(); label++) {
                reservoirSamplers[label] = ReservoirSampler<PixelInstance>(samplesPerClass);
            }

            const int randMax = 0x7FFFFFFF;
            RandomSource randomSource(randomSeed);
            Sampler sampler = randomSource.uniformSampler(randMax);

            for(unsigned int imageNr = range.begin(); imageNr != range.end(); imageNr++) {

                const auto image = trainLabelImages[imageNr].getLabelImage();
                for(int y=0; y < image.getHeight(); y++) {
                    for(int x=0; x < image.getWidth(); x++) {
                        const LabelType label = image.getLabel(x, y);

                        if (labelsToIgnore.find(label) != labelsToIgnore.end()) {
                            continue;
                        }

                        PixelInstance sample(&(trainLabelImages[imageNr].getRGBDImage()), label, x, y, horFlipSetting);
                        if (!sample.getDepth().isValid()) {
                            continue;
                        }

                        reservoirSamplers[label].sample(sampler, sample);
                    }
                }

            }

            samplersPerLabel.push_back(reservoirSamplers);

        });

    const int randMax = 0x7FFFFFFF;
    Sampler sampler = randomSource.uniformSampler(randMax);

    for (LabelType label = 0; label < classLabelPriorDistribution.size(); label++) {

        ReservoirSampler<PixelInstance> reservoirSampler(samplesPerClass);

        for (auto& samplers : samplersPerLabel) {
            const auto& samples = samplers[label].getReservoir();

            if (labelsToIgnore.find(label) != labelsToIgnore.end()) {
                if (!samples.empty()) {
                    throw std::runtime_error("sampled pixels for ignored class. must not happen");
                }
                continue;
            }

            for (const PixelInstance& sample : samples) {
                reservoirSampler.sample(sampler, sample);
            }
        }

        const std::vector<PixelInstance>& reservoir = reservoirSampler.getReservoir();

        auto color = LabelImage::decodeLabel(label);
        CURFIL_INFO((boost::format("sampled %d pixels of class '%d' RGB(%s)")
                % reservoir.size()
                % static_cast<int>(label)
                % color.toString()).str());

        allSubsamples.insert(allSubsamples.begin(), reservoir.begin(), reservoir.end());
    }

    samplingTimer.stop();

    CURFIL_INFO("sampled " << allSubsamples.size() << " pixels for "
            << numLabels << " classes (" << samplesPerClass << " samples/class)"
            << " in " << samplingTimer.format(4));

    if (allSubsamples.size() != samplesPerClass * numLabels) {
        throw std::runtime_error("failed to sample enough pixels");
    }
    return allSubsamples;
}

}

std::ostream& operator<<(std::ostream& os,
        const curfil::RandomTreeImage& tree) {
    const auto rtree = tree.getTree();
    os << rtree->countNodes() << " nodes, "
            << rtree->countLeafNodes() << " leaves, "
            << rtree->getNumClasses() << " classes, "
            << rtree->getTreeDepth() << " levels";
    return (os);
}

std::ostream& operator<<(std::ostream& os,
        const curfil::ImageFeatureFunction& featureFunction) {
    os << "ImageFeatureFunction[";
    os << "type: " << featureFunction.getTypeString();
    os << ", offset1: " << featureFunction.getOffset1();
    os << ", region1: " << featureFunction.getRegion1();
    os << ", channel1: " << static_cast<int>(featureFunction.getChannel1());
    os << ", offset2: " << featureFunction.getOffset2();
    os << ", region2: " << featureFunction.getRegion2();
    os << ", channel2: " << static_cast<int>(featureFunction.getChannel2());
    os << "]";
    return (os);
}

std::ostream& operator<<(std::ostream& os,
        const curfil::XY& xy) {
    os << "[" << xy.getX() << "," << xy.getY() << "]";
    return (os);
}
