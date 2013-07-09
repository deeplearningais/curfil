#ifndef CURFIL_PREDICT
#define CURFIL_PREDICT

#include <cuv/ndarray.hpp>
#include <string>

#include "random_tree_image_ensemble.h"

namespace curfil {

class ConfusionMatrix {

private:
    cuv::ndarray<double, cuv::host_memory_space> data;

public:
    explicit ConfusionMatrix(size_t numClasses) :
            data(numClasses, numClasses) {
        assert(numClasses > 0);
        reset();
    }

    void reset();

    void operator+=(const ConfusionMatrix& other);

    cuv::reference<double, cuv::host_memory_space>
    operator()(int label, int prediction) {
        return data(label, prediction);
    }

    double operator()(int label, int prediction) const {
        return data(label, prediction);
    }

    void increment(int label, int prediction) {
        assert(label < static_cast<int>(getNumClasses()));
        assert(prediction < static_cast<int>(getNumClasses()));
        (data(label, prediction))++;
    }

    unsigned int getNumClasses() const {
        assert(data.ndim() == 2);
        assert(data.shape(0) == data.shape(1));
        return data.shape(0);
    }

    void normalize();

    double averageClassAccuracy(bool includeVoid = true) const;

};

double calculateAccuracy(const LabelImage& image, const LabelImage* groundTruth,
        ConfusionMatrix& confusionMatrix);

double calculateAccuracyNoBackground(const LabelImage& image, const LabelImage* groundTruth);

double calculateAccuracyNoVoid(const LabelImage& image, const LabelImage* groundTruth);

void test(RandomTreeImageEnsemble& randomForest, const std::string& folderTesting,
        const std::string& folderPrediction, const double histogramBias, const bool useDepthFilling,
        const bool writeProbabilityImages, const int maxDepth);

}

std::ostream& operator<<(std::ostream& o, const curfil::ConfusionMatrix& confusionMatrix);

#endif
