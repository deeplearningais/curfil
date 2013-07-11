#ifndef CURFIL_RANDOM_FOREST_IMAGE_H
#define CURFIL_RANDOM_FOREST_IMAGE_H

#include <boost/shared_ptr.hpp>
#include <vector>

#include "random_tree_image.h"

namespace curfil {

class TreeNodes;

class RandomForestImage {
public:

    explicit RandomForestImage(const std::vector<std::string>& treeFiles,
            const std::vector<int>& deviceIds = std::vector<int>(1, 0),
            const AccelerationMode accelerationMode = GPU_ONLY,
            const double histogramBias = 0.0);

    explicit RandomForestImage(unsigned int treeCount,
            const TrainingConfiguration& configuration);

    explicit RandomForestImage(const std::vector<boost::shared_ptr<RandomTreeImage> >& ensemble,
            const TrainingConfiguration& configuration);

    void train(const std::vector<LabeledRGBDImage>& trainLabelImages, bool trainTreesSequentially = false);

    /**
     * @param image the image which should be classified
     * @param if not null, probabilities per class in a C×H×W matrix for C classes and an image of size W×H
     * @return prediction image which has the same size as 'image'
     */
    LabelImage predict(const RGBDImage& image,
            cuv::ndarray<float, cuv::host_memory_space>* prediction = 0,
            const bool onGPU = true) const;

    std::map<std::string, size_t> countFeatures() const;

    LabelType getNumClasses() const;

    const boost::shared_ptr<RandomTreeImage> getTree(size_t treeNr) const {
#ifdef NDEBUG
        return ensemble[treeNr];
#else
        return ensemble.at(treeNr);
#endif
    }

    const std::vector<boost::shared_ptr<RandomTreeImage> >& getTrees() const {
        return ensemble;
    }

    const TrainingConfiguration& getConfiguration() const {
        return configuration;
    }

    bool shouldIgnoreLabel(const LabelType& label) const;

    std::map<LabelType, RGBColor> getLabelColorMap() const;

    void normalizeHistograms(const double histogramBias);

private:

    TrainingConfiguration configuration;

    std::vector<boost::shared_ptr<RandomTreeImage> > ensemble;
    std::vector<boost::shared_ptr<const TreeNodes> > treeData;
    boost::shared_ptr<cuv::allocator> m_predictionAllocator;
};

}

std::ostream& operator<<(std::ostream& os, const curfil::RandomForestImage& ensemble);

#endif
