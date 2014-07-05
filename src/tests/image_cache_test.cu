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

#include "image.h"
#include "random_tree_image_gpu.h"
#include "test_common.h"
#include "utils.h"

BOOST_AUTO_TEST_SUITE(ImageCacheTest)

using namespace curfil;

BOOST_AUTO_TEST_CASE(testImageCacheSimple) {

    int width = 41;
    int height = 33;
    const int imageCacheSize = 3;

    std::vector<RGBDImage> images(10, RGBDImage(width, height));

    ImageCache imageCache;

    std::map<const void*, size_t>& map = imageCache.getIdMap();

    BOOST_CHECK(map.empty());

    std::vector<PixelInstance> samples;
    imageCache.copyImages(imageCacheSize, getPointers(samples));

    BOOST_CHECK(map.empty());

    samples.clear();
    samples.push_back(PixelInstance(&images[0], 0, Depth(1.0), 0, 0));
    samples.push_back(PixelInstance(&images[1], 0, Depth(1.0), 0, 0));

    imageCache.copyImages(imageCacheSize, getPointers(samples));
    BOOST_CHECK_EQUAL(2lu, map.size());

    samples.clear();
    samples.push_back(PixelInstance(&images[2], 0, Depth(1.0), 0, 0));
    samples.push_back(PixelInstance(&images[1], 0, Depth(1.0), 0, 0));

    imageCache.copyImages(imageCacheSize, getPointers(samples));
    BOOST_CHECK_EQUAL(3lu, map.size());

    samples.clear();
    samples.push_back(PixelInstance(&images[1], 0, Depth(1.0), 0, 0));
    samples.push_back(PixelInstance(&images[2], 0, Depth(1.0), 0, 0));
    samples.push_back(PixelInstance(&images[3], 0, Depth(1.0), 0, 0));

    CURFIL_INFO("now, image 0 must be dropped. image 3 must be transferred");

    imageCache.copyImages(imageCacheSize, getPointers(samples));

    BOOST_CHECK_EQUAL(3lu, map.size());
    BOOST_CHECK(map.find(&images[0]) == map.end());
    BOOST_CHECK(map.find(&images[1]) != map.end());
    BOOST_CHECK(map.find(&images[2]) != map.end());
    BOOST_CHECK(map.find(&images[3]) != map.end());

    // use image 1 and 3 but not 2, so 2 must be removed next

    samples.clear();
    samples.push_back(PixelInstance(&images[1], 0, Depth(1.0), 0, 0));
    samples.push_back(PixelInstance(&images[3], 0, Depth(1.0), 0, 0));
    samples.push_back(PixelInstance(&images[3], 0, Depth(1.0), 0, 0));
    samples.push_back(PixelInstance(&images[1], 0, Depth(1.0), 0, 0));

    imageCache.copyImages(imageCacheSize, getPointers(samples));

    // still all 3 images must be in there
    BOOST_CHECK_EQUAL(3lu, map.size());
    BOOST_CHECK(map.find(&images[0]) == map.end());
    BOOST_CHECK(map.find(&images[1]) != map.end());
    BOOST_CHECK(map.find(&images[2]) != map.end());
    BOOST_CHECK(map.find(&images[3]) != map.end());

    samples.clear();
    samples.push_back(PixelInstance(&images[4], 0, Depth(1.0), 0, 0));

    imageCache.copyImages(imageCacheSize, getPointers(samples));

    // image 2 must be evicted
    BOOST_CHECK_EQUAL(3lu, map.size());
    BOOST_CHECK(map.find(&images[0]) == map.end());
    BOOST_CHECK(map.find(&images[1]) != map.end());
    BOOST_CHECK(map.find(&images[2]) == map.end());
    BOOST_CHECK(map.find(&images[3]) != map.end());
    BOOST_CHECK(map.find(&images[4]) != map.end());

    CURFIL_INFO("done");
}
BOOST_AUTO_TEST_SUITE_END()
