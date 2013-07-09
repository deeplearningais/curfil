#ifndef CURFIL_RANDOMTREEIMAGEENSEMBLE_H
#define CURFIL_RANDOMTREEIMAGEENSEMBLE_H

#include <boost/shared_ptr.hpp>
#include <vector>

#include "random_tree_image.h"

namespace curfil {

class TreeNodes;

class RandomTreeImageEnsemble {
public:
    RandomTreeImageEnsemble(unsigned int treeCount,
            const TrainingConfiguration& configuration);

    RandomTreeImageEnsemble(const std::vector<boost::shared_ptr<RandomTreeImage> >& ensemble,
            const TrainingConfiguration& configuration);

    void train(const std::vector<LabeledRGBDImage>& trainLabelImages, bool trainTreesSequentially = false);

    cuv::ndarray<float, cuv::host_memory_space> test(const RGBDImage* image, LabelImage& prediction,
            const bool onGPU = true) const;

    void normalizeHistograms(const double histogramBias);

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

private:

    const TrainingConfiguration configuration;

    std::vector<boost::shared_ptr<RandomTreeImage> > ensemble;
    std::vector<boost::shared_ptr<const TreeNodes> > treeData;
    boost::shared_ptr<cuv::allocator> m_predictionAllocator;
};

}

std::ostream& operator<<(std::ostream& os, const curfil::RandomTreeImageEnsemble& ensemble);

#endif
