#ifndef CURFIL_PREDICT
#define CURFIL_PREDICT

#include <cuv/ndarray.hpp>
#include <string>

#include "random_forest_image.h"

namespace curfil {

/**
 * Helper class to maintain a n×n confusion matrix.
 *
 *
 * Construction of the confusion matrix is usually a two-phase process.
 *   1. Increment the label/prediction counters
 *   2. Normalize the confusion matrix such that the sum over the predictions for every class is 1.
 */
class ConfusionMatrix {

public:

    /**
     * Create an empty confusion matrix that can later be resized with resize().
     */
    explicit ConfusionMatrix() :
            data(), normalized(false) {
    }

    explicit ConfusionMatrix(const ConfusionMatrix& other) :
            data(other.data.copy()), normalized(other.normalized) {
    }

    /**
     * Create a confusion matrix of size numClasses × numClasses with zero initial values.
     */
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

    /**
     * Reset all per-class counters to zero.
     */
    void reset();

    /**
     * Resize the confusion matrix. This operation implies a reset.
     *
     * @param numClasses resize the confusion matrix to numClasses × numClasses.
     */
    void resize(unsigned int numClasses);

    /**
     * @return true if and only if the confusion matrix was normalized.
     * @see normalize()
     */
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

    /**
     * Increment the counter in the matrix for the given label and the prediction.
     * Increment can only be used <b>before</b> the confusion matrix is normalized!
     */
    void increment(int label, int prediction) {
        if (normalized) {
            throw std::runtime_error("confusion matrix is already normalized");
        }
        assert(label < static_cast<int>(getNumClasses()));
        assert(prediction < static_cast<int>(getNumClasses()));
        (data(label, prediction))++;
    }

    /**
     * @return n for a confusion matrix of size n×n
     */
    unsigned int getNumClasses() const {
        assert(data.ndim() == 2);
        assert(data.shape(0) == data.shape(1));
        return data.shape(0);
    }

    /**
     * Normalizes the confusion matrix such that the sum of the predictions equals one for every label (class).
     */
    void normalize();

    /**
     * Calculate the average per-class accuracy which is equal to the average over the diagonal.
     *
     * @param includeVoid if false, the class 0 is excluded in the calculation.
     */
    double averageClassAccuracy(bool includeVoid = true) const;

private:
    cuv::ndarray<double, cuv::host_memory_space> data;
    bool normalized;

};

/**
 * Calculates the average pixel accuracy on 'prediction' according to the ground truth.
 * Predictions where the ground truth contains 'void' (black pixels) are not counted if 'includeVoid' is set to false.
 * The confusion matrix is stored to 'confusionMatrix' if not NULL.
 */
double calculatePixelAccuracy(const LabelImage& prediction, const LabelImage& groundTruth,
        const bool includeVoid = true, ConfusionMatrix* confusionMatrix = 0);

/**
 * Helper method to run a prediction for all images in the given test folder for the given trained random forest.
 *
 * @param randomForest the random forest that was obtained by a training or loaded from disk.
 * @param folderTesting the folder that contains test images.
 * @param folderPrediction the folder where to store the prediction images to
 * @param useDepthFilling whether to run simple depth filling before prediction
 * @param writeProbabilityImages whether to store per-class probability images in the prediction folder
 */
void test(RandomForestImage& randomForest, const std::string& folderTesting,
        const std::string& folderPrediction, const bool useDepthFilling,
        const bool writeProbabilityImages);

}

std::ostream& operator<<(std::ostream& o, const curfil::ConfusionMatrix& confusionMatrix);

#endif
