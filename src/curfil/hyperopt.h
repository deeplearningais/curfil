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

/**
 * Class that stores the results of a hyperopt run
 * @ingroup hyperopt
 */
class Result {

private:
    ConfusionMatrix confusionMatrix;
    double pixelAccuracy;
    double pixelAccuracyWithoutVoid;
    LossFunctionType lossFunctionType;
    int randomSeed;

public:
    /**
     * create an instance of prediction result, including the confusion matrix and the different losses
     */
    Result(const ConfusionMatrix& confusionMatrix, double pixelAccuracy,
            double pixelAccuracyWithoutVoid, const LossFunctionType lossFunctionType) :
            confusionMatrix(confusionMatrix),
                    pixelAccuracy(pixelAccuracy),
                    pixelAccuracyWithoutVoid(pixelAccuracyWithoutVoid),
                    lossFunctionType(lossFunctionType),
                    randomSeed(0) {
    }

    /**
     * @return the BSON representation of the Result object
     */
    mongo::BSONObj toBSON() const;

    /**
     * @return the computed confusion matrix
     */
    const ConfusionMatrix& getConfusionMatrix() const {
        return confusionMatrix;
    }

    /**
     * set the loss type, can be pixel or class accuracy, with or without void
     */
    void setLossFunctionType(const LossFunctionType& lossFunctionType) {
        this->lossFunctionType = lossFunctionType;
    }

    /**
     * @return loss value (1 - accuracy), the value depends on the loss function type
     */
    double getLoss() const;

    /**
     * @return the average class accuracy including void
     */
    double getClassAccuracy() const {
        return confusionMatrix.averageClassAccuracy(true);
    }

    /**
     * @return the average class accuracy excluding void and ignored colors
     */
    double getClassAccuracyWithoutVoid() const {
        return confusionMatrix.averageClassAccuracy(false);
    }

    /**
     * @return the overall pixel accuracy including void
     */
    double getPixelAccuracy() const {
        return pixelAccuracy;
    }

    /**
     * @return the overall pixel accuracy excluding void and ignored colors
     */
    double getPixelAccuracyWithoutVoid() const {
        return pixelAccuracyWithoutVoid;
    }

    /**
     * save the random seed thatt was used in training
     */
    void setRandomSeed(int randomSeed) {
        this->randomSeed = randomSeed;
    }

    /**
     * @return the random seed stored
     */
    int getRandomSeed() const {
        return randomSeed;
    }

};

/**
 * Client that does a hyperopt parameter search
 * @ingroup hyperopt
 */
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
    bool horizontalFlipping;
    size_t numLabels;
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

    /**
     * create a hyperopt client using the parameters provided by the user
     */
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
            bool horizontalFlipping,
            size_t numLabels,
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
                    horizontalFlipping(horizontalFlipping),
                    numLabels(numLabels),
                    lossFunction(parseLossFunction(lossFunction))
    {
    }

    /**
     * do a number of train and test runs using the task parameters
     */
    void handle_task(const mongo::BSONObj& task);

    void run(); /**< continuously get the next task and handle it */
};

}

#endif
