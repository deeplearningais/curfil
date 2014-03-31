#define BOOST_TEST_MODULE example

#include <boost/test/included/unit_test.hpp>
#include <vector>

#include "hyperopt.h"
#include "test_common.h"
#include "utils.h"

using namespace curfil;

BOOST_AUTO_TEST_SUITE(HyperoptTest)

BOOST_AUTO_TEST_CASE(testContinueSearching) {

    std::vector<double> currentBest;
    currentBest.push_back(1.0 - 0.681);
    currentBest.push_back(1.0 - 0.686);
    currentBest.push_back(1.0 - 0.683);
    currentBest.push_back(1.0 - 0.679);

    std::vector<double> currentRun;
    currentRun.push_back(1.0 - 0.6293);
    currentRun.push_back(1.0 - 0.6212);
    currentRun.push_back(1.0 - 0.6210);

    // no need to continue searching
    BOOST_CHECK(!continueSearching(currentBest, currentRun));
    BOOST_CHECK(continueSearching(currentRun, currentBest));

    currentRun.clear();
    currentRun.push_back(1.0 - 0.20);
    currentRun.push_back(1.0 - 0.60);
    currentRun.push_back(1.0 - 0.65);

    // we do not know anything (high variance). continue searching
    BOOST_CHECK(continueSearching(currentBest, currentRun));

    for (int i = 0; i < 20; i++) {
        currentRun.push_back(1.0 - 0.55);
    }
    // okay. stop
    BOOST_CHECK(!continueSearching(currentBest, currentRun));

    currentRun.clear();
    currentRun.push_back(1.0 - 0.20);
    currentRun.push_back(1.0 - 0.21);
    currentRun.push_back(1.0 - 0.215);
    currentRun.push_back(1.0 - 0.912);

    // we are pretty bad. but there is one outlier which is very good. continue searching
    BOOST_CHECK(continueSearching(currentBest, currentRun));

    currentRun.clear();
    currentRun.push_back(1.0 - 0.20);
    currentRun.push_back(1.0 - 0.21);

    // too few samples. continue searching
    BOOST_CHECK(continueSearching(currentBest, currentRun));

    currentRun.clear();
    currentRun.push_back(1.0 - 0.659);
    currentRun.push_back(1.0 - 0.674);
    currentRun.push_back(1.0 - 0.672);

    // not better so far. but who knows if we get better. continue searching.
    BOOST_CHECK(continueSearching(currentBest, currentRun));

    currentRun.push_back(1.0 - 0.678);

    BOOST_CHECK(continueSearching(currentBest, currentRun));

    currentRun.push_back(1.0 - 0.671);

    BOOST_CHECK(continueSearching(currentBest, currentRun));

    currentRun.push_back(1.0 - 0.673);

    BOOST_CHECK(!continueSearching(currentBest, currentRun));
}

BOOST_AUTO_TEST_CASE(testResultToBSON) {
    ConfusionMatrix confusionMatrix(3);
    confusionMatrix(0, 0) = 1.0;
    confusionMatrix(0, 1) = 0.0;
    confusionMatrix(0, 2) = 0.0;
    confusionMatrix(1, 0) = 0.0;
    confusionMatrix(1, 1) = 0.5;
    confusionMatrix(1, 2) = 0.5;
    confusionMatrix(2, 0) = 0.0;
    confusionMatrix(2, 1) = 0.25;
    confusionMatrix(2, 2) = 0.75;

    double pixelAccuracy = 0.95;
    double pixelAccuracyWithoutVoid = 0.85;

    Result result(confusionMatrix, pixelAccuracy, pixelAccuracyWithoutVoid, LossFunctionType::CLASS_ACCURACY);
    result.setRandomSeed(4711);

    const mongo::BSONObj obj = result.toBSON();
    CURFIL_INFO(obj.jsonString(mongo::JsonStringFormat::Strict, 1));

    BOOST_REQUIRE_EQUAL(obj.getIntField("randomSeed"), 4711);
    BOOST_REQUIRE_EQUAL(obj.getField("loss").Double(), 0.25);
    BOOST_REQUIRE_EQUAL(obj.getField("classAccuracy").Double(), 0.75);
    BOOST_REQUIRE_EQUAL(obj.getField("pixelAccuracy").Double(), pixelAccuracy);
    BOOST_REQUIRE_EQUAL(obj.getField("pixelAccuracyWithoutVoid").Double(), pixelAccuracyWithoutVoid);
    BOOST_REQUIRE_EQUAL(obj.getField("confusionMatrix").toString(false),
            "[ [ 1.0, 0.0, 0.0 ], [ 0.0, 0.5, 0.5 ], [ 0.0, 0.25, 0.75 ] ]");
}

BOOST_AUTO_TEST_CASE(testResultGetLoss) {

    std::vector<LabelType> ignoredLabels;
    ignoredLabels.push_back(0);

    ConfusionMatrix confusionMatrix(3, ignoredLabels);
    confusionMatrix(0, 0) = 1.0;
    confusionMatrix(0, 1) = 0.0;
    confusionMatrix(0, 2) = 0.0;
    confusionMatrix(1, 0) = 0.0;
    confusionMatrix(1, 1) = 0.5;
    confusionMatrix(1, 2) = 0.5;
    confusionMatrix(2, 0) = 0.0;
    confusionMatrix(2, 1) = 0.25;
    confusionMatrix(2, 2) = 0.75;

    double pixelAccuracy = 0.95;
    double pixelAccuracyWithoutVoid = 0.85;

    Result result(confusionMatrix, pixelAccuracy, pixelAccuracyWithoutVoid, LossFunctionType::CLASS_ACCURACY);

    BOOST_CHECK_EQUAL(result.getLoss(), 1.0 - (1.0 + 0.5 + 0.75) / 3.0);

    result.setLossFunctionType(LossFunctionType::CLASS_ACCURACY_WITHOUT_VOID);
    BOOST_CHECK_EQUAL(result.getLoss(), 1.0 - (0.5 + 0.75) / 2.0);

    result.setLossFunctionType(LossFunctionType::PIXEL_ACCURACY);
    BOOST_CHECK_EQUAL(result.getLoss(), 1.0 - pixelAccuracy);

    result.setLossFunctionType(LossFunctionType::PIXEL_ACCURACY_WITHOUT_VOID);
    BOOST_CHECK_EQUAL(result.getLoss(), 1.0 - pixelAccuracyWithoutVoid);

}
BOOST_AUTO_TEST_SUITE_END()
