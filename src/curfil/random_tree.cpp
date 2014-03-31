#include "random_tree.h"

#include <boost/algorithm/string/join.hpp>
#include <boost/format.hpp>

namespace curfil {

namespace detail {
bool isScoreBetter(const ScoreType bestScore, const ScoreType score, const int featureNr) {
    double diff = fabs(score - bestScore);

#ifndef NDEBUG
    if (diff > 0 && diff < 1e-8) {
        CURFIL_WARNING(boost::format("close scores: %.15f (feat %d)") % bestScore % featureNr
                << boost::format(" vs %.15f, diff: %.15f") % score % diff);
    }
#endif

    if (diff < 1e-13) {
        // hack to make CPU and GPU results comparable
        return false;
    }

    return (score > bestScore);
}

cuv::ndarray<double, cuv::host_memory_space> normalizeHistogram(
        const cuv::ndarray<WeightType, cuv::host_memory_space>& histogram,
        const cuv::ndarray<WeightType, cuv::host_memory_space>& priorDistribution,
        const double histogramBias) {

    const int numLabels = histogram.size();

    cuv::ndarray<double, cuv::host_memory_space> normalizedHistogram(numLabels);

    std::vector<double> labelWeights(numLabels);
    {
        double sum = 0;
        for (int i = 0; i < numLabels; i++) {
            sum += priorDistribution[i];
        }
        for (int i = 0; i < numLabels; i++) {
            labelWeights[i] = priorDistribution[i] / sum;
        }
    }

    double sum = 0;
    for (int i = 0; i < numLabels; i++) {
        assert(sum < std::numeric_limits<WeightType>::max() - histogram[i]);
        sum += histogram[i];
    }

    for (int i = 0; i < numLabels; i++) {
        normalizedHistogram[i] = histogram[i] - histogramBias * sum;
        if (normalizedHistogram[i] < 0.0) {
            normalizedHistogram[i] = 0.0;
        }
    }

    sum = 0;
    for (int i = 0; i < numLabels; i++) {
        assert(sum < std::numeric_limits<WeightType>::max() - histogram[i]);
        sum += normalizedHistogram[i];
    }

    if (sum > 0) {
        for (int i = 0; i < numLabels; i++) {
          //  normalizedHistogram[i] = normalizedHistogram[i] / sum * labelWeights[i];
            normalizedHistogram[i] = normalizedHistogram[i] / sum ;
        }
    }

    return normalizedHistogram;
}
}

int Sampler::getNext() {
    int value = distribution(rng);
    assert(value >= lower);
    assert(value <= upper);
    return value;
}

AccelerationMode TrainingConfiguration::parseAccelerationModeString(const std::string& modeString) {
    if (modeString == "cpu") {
        return AccelerationMode::CPU_ONLY;
    } else if (modeString == "gpu") {
        return AccelerationMode::GPU_ONLY;
    } else if (modeString == "compare") {
        return AccelerationMode::GPU_AND_CPU_COMPARE;
    } else {
        throw std::runtime_error(std::string("illegal acceleration mode: ") + modeString);
    }
}

std::string TrainingConfiguration::getAccelerationModeString() const {
    switch (accelerationMode) {
        case GPU_ONLY:
            return "gpu";
        case CPU_ONLY:
            return "cpu";
        case GPU_AND_CPU_COMPARE:
            return "compare";
        default:
            throw std::runtime_error(boost::str(boost::format("unknown acceleration mode: %d") % accelerationMode));
    }
}

TrainingConfiguration::TrainingConfiguration(const TrainingConfiguration& other) {
    *this = other;
}

TrainingConfiguration& TrainingConfiguration::operator=(const TrainingConfiguration& other) {
    randomSeed = other.randomSeed;
    samplesPerImage = other.samplesPerImage;
    featureCount = other.featureCount;
    minSampleCount = other.minSampleCount;
    maxDepth = other.maxDepth;
    boxRadius = other.boxRadius;
    regionSize = other.regionSize;
    thresholds = other.thresholds;
    numThreads = other.numThreads;
    maxImages = other.maxImages;
    imageCacheSize = other.imageCacheSize;
    maxSamplesPerBatch = other.maxSamplesPerBatch;
    accelerationMode = other.accelerationMode;
    useCIELab = other.useCIELab;
    useDepthFilling = other.useDepthFilling;
    deviceIds = other.deviceIds;
    subsamplingType = other.subsamplingType;
    ignoredColors = other.ignoredColors;
    useDepthImages = other.useDepthImages;
    horizontalFlipping = other.horizontalFlipping;
    assert(*this == other);
    return *this;
}

bool TrainingConfiguration::equals(const TrainingConfiguration& other,
        bool strict) const {

    if (strict && randomSeed != other.randomSeed)
        return false;
    if (strict && deviceIds != other.deviceIds)
        return false;
    if (strict && imageCacheSize != other.imageCacheSize)
        return false;
    if (strict && maxSamplesPerBatch != other.maxSamplesPerBatch)
        return false;

    if (samplesPerImage != other.samplesPerImage)
        return false;
    if (featureCount != other.featureCount)
        return false;
    if (minSampleCount != other.minSampleCount)
        return false;
    if (maxDepth != other.maxDepth)
        return false;
    if (boxRadius != other.boxRadius)
        return false;
    if (regionSize != other.regionSize)
        return false;
    if (thresholds != other.thresholds)
        return false;
    if (numThreads != other.numThreads)
        return false;
    if (maxImages != other.maxImages)
        return false;
    if (accelerationMode != other.accelerationMode)
        return false;
    if (subsamplingType != other.subsamplingType)
        return false;
    if (ignoredColors != other.ignoredColors)
        return false;
    if (useCIELab != other.useCIELab)
        return false;
    if (useDepthFilling != other.useDepthFilling)
        return false;
    if (useDepthImages != other.useDepthImages)
    	return false;
    if (horizontalFlipping != other.horizontalFlipping)
    	return false;

    return true;
}

}

template<class T>
static std::string joinToString(const std::vector<T>& input, const std::string separator = ", ",
        const std::string prefix = "[", const std::string postfix = "]") {
    std::vector<std::string> strings;
    for (const auto& v : input) {
        strings.push_back(boost::lexical_cast<std::string>(v));
    }
    return prefix + boost::algorithm::join(strings, separator) + postfix;
}

std::ostream& operator<<(std::ostream& os, const curfil::TrainingConfiguration& configuration) {
    os << "randomSeed: " << configuration.getRandomSeed() << std::endl;
    os << "samplesPerImage: " << configuration.getSamplesPerImage() << std::endl;
    os << "featureCount: " << configuration.getFeatureCount() << std::endl;
    os << "minSampleCount: " << configuration.getMinSampleCount() << std::endl;
    os << "maxDepth: " << configuration.getMaxDepth() << std::endl;
    os << "boxRadius: " << configuration.getBoxRadius() << std::endl;
    os << "regionSize: " << configuration.getRegionSize() << std::endl;
    os << "thresholds: " << configuration.getThresholds() << std::endl;
    os << "maxImages: " << configuration.getMaxImages() << std::endl;
    os << "imageCacheSize: " << configuration.getImageCacheSize() << std::endl;
    os << "accelerationMode: " << configuration.getAccelerationModeString() << std::endl;
    os << "maxSamplesPerBatch: " << configuration.getMaxSamplesPerBatch() << std::endl;
    os << "subsamplingType: " << configuration.getSubsamplingType() << std::endl;
    os << "useCIELab: " << configuration.isUseCIELab() << std::endl;
    os << "useDepthFilling: " << configuration.isUseDepthFilling() << std::endl;
    os << "deviceIds: " << joinToString(configuration.getDeviceIds()) << std::endl;
    os << "ignoredColors: " << joinToString(configuration.getIgnoredColors()) << std::endl;
    os << "useDepthImages: " << configuration.isUseDepthImages() << std::endl;
    os << "horizontalFlipping: " << configuration.doHorizontalFlipping() << std::endl;
    return os;
}
