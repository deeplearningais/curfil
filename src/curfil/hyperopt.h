#ifndef CURFIL_HYPEROPT
#define CURFIL_HYPEROPT

#include <boost/asio/io_service.hpp>
#include <mdbq/client.hpp>
#include <mongo/bson/bson.h>

#include "image.h"
#include "predict.h"
#include "random_forest_image.h"
#include "random_tree_image.h"

namespace curfil {

bool continueSearching(const std::vector<double>& currentBestAccuracies,
        const std::vector<double>& currentRunAccuracies);

enum LossFunctionType {
    CLASS_ACCURACY, //
    CLASS_ACCURACY_WITHOUT_VOID, //
    PIXEL_ACCURACY, //
    PIXEL_ACCURACY_WITHOUT_VOID //
};

class Result {

private:
    ConfusionMatrix confusionMatrix;
    double pixelAccuracy;
    double pixelAccuracyWithoutVoid;
    LossFunctionType lossFunctionType;
    int randomSeed;

public:
    Result(const ConfusionMatrix& confusionMatrix, double pixelAccuracy,
            double pixelAccuracyWithoutVoid, const LossFunctionType lossFunctionType) :
            confusionMatrix(confusionMatrix),
                    pixelAccuracy(pixelAccuracy),
                    pixelAccuracyWithoutVoid(pixelAccuracyWithoutVoid),
                    lossFunctionType(lossFunctionType),
                    randomSeed(0) {
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

    double getClassAccuracyWithoutVoid() const {
        return confusionMatrix.averageClassAccuracy(false);
    }

    double getPixelAccuracy() const {
        return pixelAccuracy;
    }

    double getPixelAccuracyWithoutVoid() const {
        return pixelAccuracyWithoutVoid;
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
    int imageCacheSizeMB;
    int randomSeed;
    int numThreads;
    std::string subsamplingType;
    std::vector<std::string> ignoredColors;
    bool useDepthImages;
    LossFunctionType lossFunction;

    boost::asio::io_service ios;

    RandomForestImage train(size_t trees,
            const TrainingConfiguration& configuration,
            const std::vector<LabeledRGBDImage>& trainImages);

    void randomSplit(const int randomSeed, const double testRatio,
            std::vector<LabeledRGBDImage>& trainImages,
            std::vector<LabeledRGBDImage>& testImages);

    double measureTrueLoss(unsigned int numTrees, TrainingConfiguration configuration,
            const double histogramBias, double& variance);

    const Result test(const RandomForestImage& randomForest,
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
            int imageCacheSizeMB,
            int randomSeed,
            int numThreads,
            const std::string& subsamplingType,
            const std::vector<std::string>& ignoredColors,
            bool useDepthImages,
            const std::string& lossFunction,
            const std::string& url, const std::string& db, const mongo::BSONObj& jobSelector) :
            Client(url, db, jobSelector),
                    allRGBDImages(allRGBDImages),
                    allTestImages(allTestImages),
                    useCIELab(useCIELab),
                    useDepthFilling(useDepthFilling),
                    deviceIds(deviceIds),
                    maxImages(maxImages),
                    imageCacheSizeMB(imageCacheSizeMB),
                    randomSeed(randomSeed),
                    numThreads(numThreads),
                    subsamplingType(subsamplingType),
                    ignoredColors(ignoredColors),
		    useDepthImages(useDepthImages),
                    lossFunction(parseLossFunction(lossFunction))
    {
    }

    void handle_task(const mongo::BSONObj& task);

    void run();
};

}

#endif
