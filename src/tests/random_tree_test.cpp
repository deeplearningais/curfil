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

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/test/included/unit_test.hpp>
#include <cmath>

#include "random_tree.h"
#include "test_common.h"

using namespace curfil;
using namespace cuv;

BOOST_AUTO_TEST_SUITE(RandomTreeTest)

BOOST_AUTO_TEST_CASE(testNormalizeHistogramEqualPriorDistribution) {

    const LabelType NUM_LABELS = 3;

    cuv::ndarray<WeightType, cuv::host_memory_space> histogram(NUM_LABELS);
    histogram[0] = 850;
    histogram[1] = 50;
    histogram[2] = 100;

    cuv::ndarray<WeightType, cuv::host_memory_space> priorDistribution(NUM_LABELS);
    for (LabelType label = 0; label < NUM_LABELS; label++) {
        priorDistribution[label] = 100;
    }

    ndarray<double, host_memory_space> normalizedHistogram = curfil::detail::normalizeHistogram(histogram,
            priorDistribution, 0.0);

    BOOST_CHECK_CLOSE(static_cast<double>(normalizedHistogram[0]), 0.850 / 3.0, 1e-15);
    BOOST_CHECK_CLOSE(static_cast<double>(normalizedHistogram[1]), 0.050 / 3.0, 1e-15);
    BOOST_CHECK_CLOSE(static_cast<double>(normalizedHistogram[2]), 0.100 / 3.0, 1e-15);
}

BOOST_AUTO_TEST_CASE(testNormalizeHistogramEqualHistogramUnequalPriorDistribution) {

    const LabelType NUM_LABELS = 3;

    cuv::ndarray<WeightType, cuv::host_memory_space> histogram(NUM_LABELS);
    histogram[0] = 50;
    histogram[1] = 50;
    histogram[2] = 50;

    cuv::ndarray<WeightType, cuv::host_memory_space> priorDistribution(NUM_LABELS);
    priorDistribution[0] = 80;
    priorDistribution[1] = 10;
    priorDistribution[2] = 10;

    ndarray<double, host_memory_space> normalizedHistogram = curfil::detail::normalizeHistogram(histogram,
            priorDistribution, 0.0);

    BOOST_CHECK_CLOSE(static_cast<double>(normalizedHistogram[0]), (1 / 3.0) * 0.8, 1e-15);
    BOOST_CHECK_CLOSE(static_cast<double>(normalizedHistogram[1]), (1 / 3.0) * 0.1, 1e-15);
    BOOST_CHECK_CLOSE(static_cast<double>(normalizedHistogram[2]), (1 / 3.0) * 0.1, 1e-15);
}

BOOST_AUTO_TEST_CASE(testNormalizeHistogramHighBias) {

    const LabelType NUM_LABELS = 3;

    cuv::ndarray<WeightType, cuv::host_memory_space> histogram(NUM_LABELS);
    histogram[0] = 50;
    histogram[1] = 50;
    histogram[2] = 50;

    cuv::ndarray<WeightType, cuv::host_memory_space> priorDistribution(NUM_LABELS);
    priorDistribution[0] = 80;
    priorDistribution[1] = 10;
    priorDistribution[2] = 10;

    ndarray<double, host_memory_space> normalizedHistogram = curfil::detail::normalizeHistogram(histogram,
            priorDistribution, 0.5);

    BOOST_CHECK_CLOSE(static_cast<double>(normalizedHistogram[0]), 0.0, 1e-15);
    BOOST_CHECK_CLOSE(static_cast<double>(normalizedHistogram[1]), 0.0, 1e-15);
    BOOST_CHECK_CLOSE(static_cast<double>(normalizedHistogram[2]), 0.0, 1e-15);
}

BOOST_AUTO_TEST_CASE(testNormalizeHistogramMediumBias) {

    const LabelType NUM_LABELS = 3;

    cuv::ndarray<WeightType, cuv::host_memory_space> histogram(NUM_LABELS);
    histogram[0] = 60;
    histogram[1] = 20;
    histogram[2] = 20;

    cuv::ndarray<WeightType, cuv::host_memory_space> priorDistribution(NUM_LABELS);
    priorDistribution[0] = 50;
    priorDistribution[1] = 25;
    priorDistribution[2] = 25;

    ndarray<double, host_memory_space> normalizedHistogram = curfil::detail::normalizeHistogram(histogram,
            priorDistribution, 0.5);

    BOOST_CHECK_CLOSE(static_cast<double>(normalizedHistogram[0]), 0.5, 1e-15);
    BOOST_CHECK_CLOSE(static_cast<double>(normalizedHistogram[1]), 0.0, 1e-15);
    BOOST_CHECK_CLOSE(static_cast<double>(normalizedHistogram[2]), 0.0, 1e-15);
}

BOOST_AUTO_TEST_CASE(testNormalizeHistogramLowBias) {

    const LabelType NUM_LABELS = 3;

    cuv::ndarray<WeightType, cuv::host_memory_space> histogram(NUM_LABELS);
    histogram[0] = 20;
    histogram[1] = 40;
    histogram[2] = 40;

    cuv::ndarray<WeightType, cuv::host_memory_space> priorDistribution(NUM_LABELS);
    priorDistribution[0] = 20;
    priorDistribution[1] = 10;
    priorDistribution[2] = 10;

    ndarray<double, host_memory_space> normalizedHistogram = curfil::detail::normalizeHistogram(histogram,
            priorDistribution, 0.2);

    BOOST_CHECK_CLOSE(static_cast<double>(normalizedHistogram[0]), 0.0, 1e-15);
    BOOST_CHECK_CLOSE(static_cast<double>(normalizedHistogram[1]), 0.5 * 0.25, 1e-15);
    BOOST_CHECK_CLOSE(static_cast<double>(normalizedHistogram[2]), 0.5 * 0.25, 1e-15);
}

BOOST_AUTO_TEST_CASE(testReservoirSampler) {

    size_t sampleSize = 1000;
    const int MAX = 100000;

    RandomSource randomSource(4711);
    Sampler sampler = randomSource.uniformSampler(0, 10 * MAX);

    ReservoirSampler<int> reservoirSampler(sampleSize);
    for (int i = 0; i < MAX; i++) {
        reservoirSampler.sample(sampler, i);
    }

    const auto& reservoir = reservoirSampler.getReservoir();
    BOOST_REQUIRE_EQUAL(reservoir.size(), sampleSize);

    boost::accumulators::accumulator_set<int,
            boost::accumulators::features<
                    boost::accumulators::tag::min,
                    boost::accumulators::tag::max,
                    boost::accumulators::tag::mean,
                    boost::accumulators::tag::variance> > acc;

    acc = std::for_each(reservoir.begin(), reservoir.end(), acc);

    double min = boost::accumulators::min(acc);
    double max = boost::accumulators::max(acc);
    double mean = boost::accumulators::mean(acc);
    double stddev = std::sqrt(static_cast<double>(boost::accumulators::variance(acc)));

    /*
     * values are empirically determined with python and numpy
     *
     * >>> import random, numpy as np
     * >>> a = np.arange(100000)
     * >>> random.shuffle(a)
     * >>> b = a[:10000]
     * >>> b.mean()
     * 50271.870999999999
     *
     * >>> np.sqrt(b.var())
     * 28804.962206705768
     *
     * >>> b.min()
     * 12.0
     *
     * >>> b.max()
     * 99996.0
     */
    BOOST_REQUIRE_LT(min, 0.01 * MAX);
    BOOST_REQUIRE_GT(max, 0.99 * MAX);
    BOOST_REQUIRE_CLOSE(mean, 50000.0, 0.05);
    BOOST_REQUIRE_CLOSE(stddev, 28000.0, 2.0);
}
BOOST_AUTO_TEST_SUITE_END()
