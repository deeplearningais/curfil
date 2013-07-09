#ifndef CURFIL_HYPEROPT
#define CURFIL_HYPEROPT

#include <boost/asio/io_service.hpp>
#include <mdbq/client.hpp>
#include <mongo/bson/bson.h>

#include "image.h"
#include "predict.h"
#include "random_tree_image_ensemble.h"
#include "random_tree_image.h"

namespace curfil {

bool continueSearching(const std::vector<double>& currentBestAccuracies,
        const std::vector<double>& currentRunAccuracies);

enum LossFunctionType {
    CLASS_ACCURACY, //
    CLASS_ACCURACY_NO_VOID, //
    PIXEL_ACCURACY, //
    PIXEL_ACCURACY_NO_BACKGROUND, //
    PIXEL_ACCURACY_NO_VOID //
};

class Result {

private:
    ConfusionMatrix confusionMatrix;
    double pixelAccuracy;
    double pixelAccuracyNoBg;
    double pixelAccuracyNoVoid;
    LossFunctionType lossFunctionType;
    int randomSeed;

public:
    Result(const ConfusionMatrix& confusionMatrix, double pixelAccuracy,
            double pixelAccuracyNoBg, double pixelAccuracyNoVoid, const LossFunctionType lossFunctionType) :
            confusionMatrix(confusionMatrix),
                    pixelAccuracy(pixelAccuracy),
                    pixelAccuracyNoBg(pixelAccuracyNoBg),
                    pixelAccuracyNoVoid(pixelAccuracyNoVoid),
                    lossFunctionType(lossFunctionType),
                    randomSeed(0) {

        this->confusionMatrix.normalize();
    }

    mongo::BSONObj toBSON() const;

    const ConfusionMatrix& getConfusionMatrix() const {
        return confusionMatrix;
    }

    void setLossFunctionType(const LossFunctionType& lossFunctionType) {
        this->lossFunctionType = lossFunctionType;
    }

    double getLoss() const;

    double getClassAccuracy() const {
        return confusionMatrix.averageClassAccuracy(true);
    }

    double getClassAccuracyNoVoid() const {
        return confusionMatrix.averageClassAccuracy(false);
    }

    double getPixelAccuracy() const {
        return pixelAccuracy;
    }

    double getPixelAccuracyNoBackground() const {
        return pixelAccuracyNoBg;
    }

    double getPixelAccuracyNoVoid() const {
        return pixelAccuracyNoVoid;
    }

    void setRandomSeed(int randomSeed) {
        this->randomSeed = randomSeed;
    }

    int getRandomSeed() const {
        return randomSeed;
    }

};

class HyperoptClient: public mdbq::Client {

private:

    const std::vector<LabeledRGBDImage>& allRGBDImages;
    const std::vector<LabeledRGBDImage>& allTestImages;

    bool useCIELab;
    bool useDepthFilling;
    std::vector<int> deviceIds;
    int maxImages;
    int imageCacheSize;
    int maxSamplesPerBatch;
    int randomSeed;
    int numThreads;
    std::string subsamplingType;
    std::vector<std::string> ignoredColors;
    LossFunctionType lossFunction;

    boost::asio::io_service ios;

    RandomTreeImageEnsemble train(size_t trees,
            const TrainingConfiguration& configuration,
            const std::vector<LabeledRGBDImage>& trainImages);

    void randomSplit(const int randomSeed, const double testRatio,
            std::vector<LabeledRGBDImage>& trainImages,
            std::vector<LabeledRGBDImage>& testImages);

    double measureTrueLoss(unsigned int numTrees, TrainingConfiguration configuration,
            const double histogramBias, double& variance);

    const Result test(const RandomTreeImageEnsemble& randomForest,
            const std::vector<LabeledRGBDImage>& testImages);

    double getParameterDouble(const mongo::BSONObj& task, const std::string& field);

    static double getAverageLossAndVariance(const std::vector<Result>& results, double& variance);

    static LossFunctionType parseLossFunction(const std::string& lossFunction);

public:

    HyperoptClient(
            const std::vector<LabeledRGBDImage>& allRGBDImages,
            const std::vector<LabeledRGBDImage>& allTestImages,
            bool useCIELab,
            bool useDepthFilling,
            const std::vector<int>& deviceIds,
            int maxImages,
            int imageCacheSize,
            int maxSamplesPerBatch,
            int randomSeed,
            int numThreads,
            const std::string& subsamplingType,
            const std::vector<std::string>& ignoredColors,
            const std::string& lossFunction,
            const std::string& url, const std::string& db, const mongo::BSONObj& jobSelector) :
            Client(url, db, jobSelector),
                    allRGBDImages(allRGBDImages),
                    allTestImages(allTestImages),
                    useCIELab(useCIELab),
                    useDepthFilling(useDepthFilling),
                    deviceIds(deviceIds),
                    maxImages(maxImages),
                    imageCacheSize(imageCacheSize),
                    maxSamplesPerBatch(maxSamplesPerBatch),
                    randomSeed(randomSeed),
                    numThreads(numThreads),
                    subsamplingType(subsamplingType),
                    ignoredColors(ignoredColors),
                    lossFunction(parseLossFunction(lossFunction))
    {
    }

    void handle_task(const mongo::BSONObj& task);

    void run();
};

}

#endif
