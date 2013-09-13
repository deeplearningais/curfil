#include "random_forest_image.h"

#include <boost/shared_ptr.hpp>
#include <cassert>
#include <tbb/parallel_for_each.h>
#include <tbb/task_scheduler_init.h>
#include <vector>

#include "image.h"
#include "import.h"
#include "random_tree_image_gpu.h"
#include "utils.h"

namespace curfil {

RandomForestImage::RandomForestImage(const std::vector<std::string>& treeFiles,
                    const std::vector<int>& deviceIds,
                    const AccelerationMode accelerationMode,
                    const double histogramBias)
 : configuration(), ensemble(treeFiles.size()),
   m_predictionAllocator(boost::make_shared<cuv::pooled_cuda_allocator>())
{

    if (treeFiles.empty()) {
        throw std::runtime_error("cannot construct empty forest");
    }

    std::vector<TrainingConfiguration> configurations(treeFiles.size());

    tbb::parallel_for(tbb::blocked_range<size_t>(0, treeFiles.size(), 1),
            [&](const tbb::blocked_range<size_t>& range) {

                for(size_t tree = range.begin(); tree != range.end(); tree++) {
                    CURFIL_INFO("reading tree " << tree << " from " << treeFiles[tree]);

                    boost::shared_ptr<RandomTreeImage> randomTree;

                    std::string hostname;
                    boost::filesystem::path folderTraining;
                    boost::posix_time::ptime date;

                    TrainingConfiguration configuration = RandomTreeImport::readJSON(treeFiles[tree], randomTree, hostname,
                            folderTraining, date);

                    CURFIL_INFO("trained " << date << " on " << hostname);
                    CURFIL_INFO("training folder: " << folderTraining);

                    assert(randomTree);

                    ensemble[tree] = randomTree;
                    configurations[tree] = configuration;

                    CURFIL_INFO(*randomTree);
                }

            });

    for (size_t i = 1; i < treeFiles.size(); i++) {
        bool strict = false;
        if (!configurations[0].equals(configurations[i], strict)) {
            CURFIL_ERROR("configuration of tree 0: " << configurations[0]);
            CURFIL_ERROR("configuration of tree " << i << ": " << configurations[i]);
            throw std::runtime_error("different configurations");
        }

        if (ensemble[0]->getTree()->getNumClasses() != ensemble[i]->getTree()->getNumClasses()) {
            CURFIL_ERROR("number of classes of tree 0: " << ensemble[0]->getTree()->getNumClasses());
            CURFIL_ERROR("number of classes of tree " << i << ": " << ensemble[i]->getTree()->getNumClasses());
            throw std::runtime_error("different number of classes in trees");
        }
    }

    CURFIL_INFO("training configuration " << configurations[0]);

    this->configuration = configurations[0];
    this->configuration.setDeviceIds(deviceIds);
    this->configuration.setAccelerationMode(accelerationMode);

    normalizeHistograms(histogramBias);
}

RandomForestImage::RandomForestImage(unsigned int treeCount, const TrainingConfiguration& configuration) :
        configuration(configuration), ensemble(treeCount),
                m_predictionAllocator(boost::make_shared<cuv::pooled_cuda_allocator>())
{
    assert(treeCount > 0);
}

RandomForestImage::RandomForestImage(const std::vector<boost::shared_ptr<RandomTreeImage> >& ensemble,
        const TrainingConfiguration& configuration) :
        configuration(configuration), ensemble(ensemble),
                m_predictionAllocator(boost::make_shared<cuv::pooled_cuda_allocator>()) {
    assert(!ensemble.empty());
#ifndef NDEBUG
    for (auto& tree : ensemble) {
        assert(tree);
        assert(tree->getTree()->isRoot());
    }
#endif

}

// Usage identical to RandomTreeImage class
void RandomForestImage::train(const std::vector<LabeledRGBDImage>& trainLabelImages,
        bool trainTreesSequentially) {

    if (trainLabelImages.empty()) {
        throw std::runtime_error("no training images");
    }

    const size_t treeCount = ensemble.size();

    const int numThreads = configuration.getNumThreads();
    tbb::task_scheduler_init init(numThreads);

    CURFIL_INFO("learning image tree ensemble. " << treeCount << " trees with " << numThreads << " threads");

    for (size_t treeNr = 0; treeNr < treeCount; ++treeNr) {
        ensemble[treeNr] = boost::make_shared<RandomTreeImage>(treeNr, configuration);
    }

    RandomSource randomSource(configuration.getRandomSeed());
    const int SEED = randomSource.uniformSampler(0xFFFF).getNext();

    auto train =
            [&](boost::shared_ptr<RandomTreeImage>& tree) {
                utils::Timer timer;
                auto seed = SEED + tree->getId();
                RandomSource randomSource(seed);

                std::vector<LabeledRGBDImage> sampledTrainLabelImages = trainLabelImages;

                if (configuration.getMaxImages() > 0 && static_cast<int>(trainLabelImages.size()) > configuration.getMaxImages()) {
                    ReservoirSampler<LabeledRGBDImage> reservoirSampler(configuration.getMaxImages());
                    Sampler sampler = randomSource.uniformSampler(0, 10 * trainLabelImages.size());
                    for (auto& image : trainLabelImages) {
                        reservoirSampler.sample(sampler, image);
                    }

                    CURFIL_INFO("tree " << tree->getId() << ": sampled " << reservoirSampler.getReservoir().size()
                            << " out of " << trainLabelImages.size() << " images");
                    sampledTrainLabelImages = reservoirSampler.getReservoir();
                }

                tree->train(sampledTrainLabelImages, randomSource, configuration.getSamplesPerImage() / treeCount);
                CURFIL_INFO("finished tree " << tree->getId() << " with random seed " << seed << " in " << timer.format(3));
            };

    if (!trainTreesSequentially && numThreads > 1) {
        tbb::parallel_for_each(ensemble.begin(), ensemble.end(), train);
    } else {
        std::for_each(ensemble.begin(), ensemble.end(), train);
    }
}

LabelImage RandomForestImage::predict(const RGBDImage& image,
         cuv::ndarray<float, cuv::host_memory_space>* probabilities, const bool onGPU, bool useDepthImages) const {

    LabelImage prediction(image.getWidth(), image.getHeight());

    const LabelType numClasses = getNumClasses();

    if (treeData.size() != ensemble.size()) {
        throw std::runtime_error((boost::format("tree data size: %d, ensemble size: %d. histograms normalized?")
                % treeData.size() % ensemble.size()).str());
    }

    cuv::ndarray<float, cuv::host_memory_space> hostProbabilities(
            cuv::extents[numClasses][image.getHeight()][image.getWidth()],
            m_predictionAllocator);

    if (onGPU) {
        cuv::ndarray<float, cuv::dev_memory_space> deviceProbabilities(
                cuv::extents[numClasses][image.getHeight()][image.getWidth()],
                m_predictionAllocator);
        cudaSafeCall(cudaMemset(deviceProbabilities.ptr(), 0, static_cast<size_t>(deviceProbabilities.size() * sizeof(float))));

        {
            utils::Profile profile("classifyImagesGPU");
            for (const boost::shared_ptr<const TreeNodes>& data : treeData) {
                classifyImage(treeData.size(), deviceProbabilities, image, numClasses, data, useDepthImages);
            }
        }

        normalizeProbabilities(deviceProbabilities);

        cuv::ndarray<LabelType, cuv::dev_memory_space> output(image.getHeight(), image.getWidth(),
                m_predictionAllocator);
        determineMaxProbabilities(deviceProbabilities, output);

        hostProbabilities = deviceProbabilities;
        cuv::ndarray<LabelType, cuv::host_memory_space> outputHost(image.getHeight(), image.getWidth(),
                m_predictionAllocator);

        outputHost = output;

        {
            utils::Profile profile("setLabels");
            for (int y = 0; y < image.getHeight(); ++y) {
                for (int x = 0; x < image.getWidth(); ++x) {
                    prediction.setLabel(x, y, static_cast<LabelType>(outputHost(y, x)));
                }
            }
        }
    } else {
        utils::Profile profile("classifyImagesCPU");

        tbb::parallel_for(tbb::blocked_range<size_t>(0, image.getHeight()),
                [&](const tbb::blocked_range<size_t>& range) {
                    for(size_t y = range.begin(); y != range.end(); y++) {
                        for(int x=0; x < image.getWidth(); x++) {

                            for (LabelType label = 0; label < numClasses; label++) {
                                hostProbabilities(label, y, x) = 0.0f;
                            }

                            for (const auto& tree : ensemble) {
                                const auto& t = tree->getTree();
                                PixelInstance pixel(&image, 0, x, y);
                                const auto& hist = t->classifySoft(pixel);
                                assert(hist.size() == numClasses);
                                for(LabelType label = 0; label<hist.size(); label++) {
                                    hostProbabilities(label, y, x) += hist[label];
                                }
                            }

                            double sum = 0.0f;
                            for (LabelType label = 0; label < numClasses; label++) {
                                sum += hostProbabilities(label, y, x);
                            }
                            float bestProb = -1.0f;
                            for (LabelType label = 0; label < numClasses; label++) {
                                hostProbabilities(label, y, x) /= sum;
                                float prob = hostProbabilities(label, y, x);
                                if (prob > bestProb) {
                                    prediction.setLabel(x, y, label);
                                    bestProb = prob;
                                }
                            }
                        }
                    }
                });
    }

    if (probabilities) {
        *probabilities = hostProbabilities;
    }

    return prediction;
}

LabelType RandomForestImage::getNumClasses() const {
    LabelType numClasses = 0;
    for (const boost::shared_ptr<RandomTreeImage>& tree : ensemble) {
        if (numClasses == 0) {
            numClasses = tree->getTree()->getNumClasses();
        } else if (tree->getTree()->getNumClasses() != numClasses) {
            throw std::runtime_error("number of classes differ in trees");
        }
    }
    return numClasses;
}

void RandomForestImage::normalizeHistograms(const double histogramBias) {

    treeData.clear();

    for (size_t treeNr = 0; treeNr < ensemble.size(); treeNr++) {
        CURFIL_INFO("normalizing histograms of tree " << treeNr <<
                " with " << ensemble[treeNr]->getTree()->countLeafNodes() << " leaf nodes");
        ensemble[treeNr]->normalizeHistograms(histogramBias);
        treeData.push_back(convertTree(ensemble[treeNr]));
    }
}

std::map<LabelType, RGBColor> RandomForestImage::getLabelColorMap() const {
    std::map<LabelType, RGBColor> labelColorMap;

    for (size_t treeNr = 0; treeNr < ensemble.size(); treeNr++) {
        const cuv::ndarray<WeightType, cuv::host_memory_space>& hist = ensemble[treeNr]->getTree()->getHistogram();
        for (LabelType label = 0; label < hist.size(); label++) {
            labelColorMap[label] = LabelImage::decodeLabel(label);
        }
    }

    return labelColorMap;
}

bool RandomForestImage::shouldIgnoreLabel(const LabelType& label) const {
    for (const auto& tree : ensemble) {
        if (tree->shouldIgnoreLabel(label)) {
            return true;
        }
    }
    return false;
}

std::map<std::string, size_t> RandomForestImage::countFeatures() const {
    std::map<std::string, size_t> featureCounts;
    for (const auto& tree : ensemble) {
        tree->getTree()->countFeatures(featureCounts);
    }
    return featureCounts;
}

}

std::ostream& operator<<(std::ostream& os, const curfil::RandomForestImage& ensemble) {
    for (const boost::shared_ptr<curfil::RandomTreeImage> tree : ensemble.getTrees()) {
        os << "   " << *(tree.get()) << std::endl;
    }
    return (os);
}
