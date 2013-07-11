#ifndef CURFIL_PREDICT
#define CURFIL_PREDICT

#include <cuv/ndarray.hpp>
#include <string>

#include "random_forest_image.h"

namespace curfil {

class ConfusionMatrix {

private:
    cuv::ndarray<double, cuv::host_memory_space> data;
    bool normalized;

public:

    explicit ConfusionMatrix() :
            data(), normalized(false) {
    }

    explicit ConfusionMatrix(const ConfusionMatrix& other) :
            data(other.data.copy()), normalized(other.normalized) {
    }

    explicit ConfusionMatrix(size_t numClasses) :
            data(numClasses, numClasses), normalized(false) {
        assert(numClasses > 0);
        reset();
    }

    ConfusionMatrix& operator=(const ConfusionMatrix& other) {
        data = other.data.copy();
        normalized = other.normalized;
        return *this;
    }

    void reset();

    void resize(unsigned int numClasses);

    bool isNormalized() const {
        return normalized;
    }

    void operator+=(const ConfusionMatrix& other);

    cuv::reference<double, cuv::host_memory_space>
    operator()(int label, int prediction) {
        return data(label, prediction);
    }

    double operator()(int label, int prediction) const {
        return data(label, prediction);
    }

    void increment(int label, int prediction) {
        if (normalized) {
            throw std::runtime_error("confusion matrix is already normalized");
        }
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

}
;

/**
 * Calculates the average pixel accuracy on 'prediction' according to the ground truth.
 * Predictions where the ground truth contains 'void' (black pixels) are not counted if 'includeVoid' is set to false.
 * The confusion matrix is stored to 'confusionMatrix' if not NULL.
 */
double calculatePixelAccuracy(const LabelImage& prediction, const LabelImage& groundTruth,
        const bool includeVoid = true, ConfusionMatrix* confusionMatrix = 0);

void test(RandomForestImage& randomForest, const std::string& folderTesting,
        const std::string& folderPrediction, const bool useDepthFilling,
        const bool writeProbabilityImages, const int maxDepth);

}

std::ostream& operator<<(std::ostream& o, const curfil::ConfusionMatrix& confusionMatrix);

#endif
