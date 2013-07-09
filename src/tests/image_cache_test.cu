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

    INFO("now, image 0 must be dropped. image 3 must be transferred");

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

    INFO("done");
}
BOOST_AUTO_TEST_SUITE_END()
