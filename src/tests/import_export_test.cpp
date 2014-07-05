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

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/test/included/unit_test.hpp>
#include <math.h>
#include <stdlib.h>
#include <tbb/task_scheduler_init.h>
#include <vector>
#include <vigra/colorconversions.hxx>
#include <vigra/copyimage.hxx>
#include <vigra/impex.hxx>
#include <vigra/stdimage.hxx>
#include <vigra/transformimage.hxx>

#include "export.h"
#include "image.h"
#include "import.h"
#include "random_forest_image.h"
#include "random_tree_image.h"
#include "test_common.h"

using namespace curfil;

static const int NUM_THREADS = 4;
static const std::string folderOutput("test.out");

BOOST_AUTO_TEST_SUITE(ImportExportTest)

template<class T, class M>
static void checkEquals(const cuv::ndarray<T, M>& a, const cuv::ndarray<T, M>& b) {
    BOOST_CHECK_EQUAL(a.size(), b.size());
    BOOST_CHECK_EQUAL(a.ndim(), b.ndim());
    for (size_t i = 0; i < a.size(); i++) {
        BOOST_CHECK_EQUAL(a[i], b[i]);
    }
}

static void checkTrees(const boost::shared_ptr<const RandomTree<PixelInstance, ImageFeatureFunction> >& a,
        const boost::shared_ptr<const RandomTree<PixelInstance, ImageFeatureFunction> >& b) {

    BOOST_CHECK_EQUAL(a->getNodeId(), b->getNodeId());
    BOOST_CHECK_EQUAL(a->getLevel(), b->getLevel());

    BOOST_CHECK_EQUAL(a->countNodes(), b->countNodes());
    BOOST_CHECK_EQUAL(a->countLeafNodes(), b->countLeafNodes());

    BOOST_CHECK_EQUAL(a->isRoot(), b->isRoot());
    BOOST_CHECK_EQUAL(a->isLeaf(), b->isLeaf());

    BOOST_CHECK_EQUAL(a->getNumClasses(), b->getNumClasses());
    BOOST_CHECK_EQUAL(a->getTreeDepth(), b->getTreeDepth());

    checkEquals(a->getHistogram(), b->getHistogram());

    if (!a->isLeaf()) {
        BOOST_CHECK_EQUAL(a->getSplit().getFeatureId(), b->getSplit().getFeatureId());
        BOOST_CHECK_CLOSE(a->getSplit().getThreshold(), b->getSplit().getThreshold(), 1e-4);
        BOOST_CHECK_CLOSE(a->getSplit().getScore(), b->getSplit().getScore(), 1e-5);

        checkTrees(a->getLeft(), b->getLeft());
        checkTrees(a->getRight(), b->getRight());
    } else {
        BOOST_CHECK(a->getLeft() == NULL);
        BOOST_CHECK(b->getLeft() == NULL);
        BOOST_CHECK(a->getRight() == NULL);
        BOOST_CHECK(b->getRight() == NULL);
    }
}
static void checkTrees(const boost::shared_ptr<const RandomTreeImage>& a,
        const boost::shared_ptr<const RandomTreeImage>& b) {
    BOOST_CHECK(a->getClassLabelPriorDistribution() == b->getClassLabelPriorDistribution());
    checkTrees(a->getTree(), b->getTree());
}

BOOST_AUTO_TEST_CASE(testExportImport) {
    std::vector<LabeledRGBDImage> trainImages;
    const bool useCIELab = true;
    const bool useDepthFilling = false;
    const bool useDepthImages = true;

    if (boost::unit_test::framework::master_test_suite().argc < 2) {
        throw std::runtime_error("please specify folder with testdata");
    }
    const std::string folderTraining(boost::unit_test::framework::master_test_suite().argv[1]);

    trainImages.push_back(loadImagePair(folderTraining + "/training1_colors.png", useCIELab, useDepthImages,useDepthFilling));

    // Train

    size_t trees = 3;

    unsigned int samplesPerImage = 500;
    unsigned int featureCount = 500;
    unsigned int minSampleCount = 32;
    int maxDepth = 10;
    uint16_t boxRadius = 50;
    uint16_t regionSize = 10;
    uint16_t thresholds = 10;
    int numThreads = NUM_THREADS;
    int maxImages = 10;
    int imageCacheSize = 10;
    unsigned int maxSamplesPerBatch = 5000;
    AccelerationMode accelerationMode = AccelerationMode::GPU_AND_CPU_COMPARE;

    const int SEED = 4711;

    std::vector<std::string> ignoredColors;
    ignoredColors.push_back("50,205,50"); // this color must exist. otherwise an exception will be thrown

    std::vector<int> deviceIds;
    deviceIds.push_back(0);

    TrainingConfiguration configuration(SEED, samplesPerImage, featureCount, minSampleCount, maxDepth, boxRadius,
            regionSize, thresholds, numThreads, maxImages, imageCacheSize, maxSamplesPerBatch, accelerationMode,
            true, false, deviceIds, "classUniform", ignoredColors);

    RandomForestImage randomForest(trees, configuration);
    randomForest.train(trainImages);

    for (const bool verbose : { true, false }) {
        boost::filesystem::create_directory(folderOutput);
        RandomTreeExport treeExport(configuration, folderOutput, folderTraining, verbose);
        treeExport.writeJSON(randomForest);

        for (size_t treeNr = 0; treeNr < trees; treeNr++) {
            const std::string filename = boost::str(boost::format("%s/tree%d.json.gz") % folderOutput % treeNr);
            boost::shared_ptr<RandomTreeImage> tree;
            std::string hostname;
            boost::filesystem::path trainingFolder;
            boost::posix_time::ptime date;
            TrainingConfiguration readConfiguration = RandomTreeImport::readJSON(filename, tree, hostname,
                    trainingFolder, date);
            BOOST_CHECK(readConfiguration == configuration);
            BOOST_CHECK(tree);
            checkTrees(tree, randomForest.getTree(treeNr));
        }
    }

}
BOOST_AUTO_TEST_SUITE_END()
