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
