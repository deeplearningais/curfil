#define BOOST_TEST_MODULE example

#include <assert.h>
#include <boost/filesystem.hpp>
#include <boost/test/included/unit_test.hpp>

#include "image.h"
#include "random_tree_image.h"
#include "utils.h"

namespace fs = boost::filesystem;

class Fixture {
public:
    Fixture() :
            temporaryColorFile(fs::unique_path("%%%%-%%%%-%%%%-%%%%.png")),
                    temporaryDepthFile(fs::unique_path("%%%%-%%%%-%%%%-%%%%_depth.png")) {
    }

    ~Fixture() {
        fs::remove(temporaryColorFile);
        fs::remove(temporaryDepthFile);
    }

    fs::path temporaryColorFile;
    fs::path temporaryDepthFile;
};

BOOST_FIXTURE_TEST_SUITE(ImageTest, Fixture)

using namespace curfil;

BOOST_AUTO_TEST_CASE(testParseColorString) {
    auto color = RGBColor("0,0000,00");
    BOOST_REQUIRE_EQUAL(color.size(), 3);
    BOOST_REQUIRE_EQUAL(color[0], 0u);
    BOOST_REQUIRE_EQUAL(color[1], 0u);
    BOOST_REQUIRE_EQUAL(color[2], 0u);

    color = RGBColor("255,0255,255");
    BOOST_REQUIRE_EQUAL(color.size(), 3);
    BOOST_REQUIRE_EQUAL(color[0], 255u);
    BOOST_REQUIRE_EQUAL(color[1], 255u);
    BOOST_REQUIRE_EQUAL(color[2], 255u);

    color = RGBColor("129,64,81");
    BOOST_REQUIRE_EQUAL(color.size(), 3);
    BOOST_REQUIRE_EQUAL(color[0], 129u);
    BOOST_REQUIRE_EQUAL(color[1], 64u);
    BOOST_REQUIRE_EQUAL(color[2], 81u);

    BOOST_CHECK_THROW(RGBColor("-1,0,0"), std::runtime_error);
    BOOST_CHECK_THROW(RGBColor("256,0,0"), std::runtime_error);
    BOOST_CHECK_THROW(RGBColor("0,0,256"), std::runtime_error);
    BOOST_CHECK_THROW(RGBColor("a,0,0"), std::runtime_error);
    BOOST_CHECK_THROW(RGBColor(""), std::runtime_error);
    BOOST_CHECK_THROW(RGBColor(",,"), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(testColorIntegral) {
    RGBDImage image(3, 3);

    for (int c = 0; c < 3; ++c) {
        image.setColor(0, 0, c, 5);
        image.setColor(1, 0, c, 2);
        image.setColor(2, 0, c, 3);
        image.setColor(0, 1, c, 1);
        image.setColor(1, 1, c, 5);
        image.setColor(2, 1, c, 4);
        image.setColor(0, 2, c, 2);
        image.setColor(1, 2, c, 2);
        image.setColor(2, 2, c, 1);
    }

    image.dump(std::cout);
    image.calculateIntegral();
    std::cout << "integral" << std::endl;
    image.dump(std::cout);

    for (int c = 0; c < 3; ++c) {
        BOOST_CHECK_EQUAL(image.getColor(0, 0, c), 5);
        BOOST_CHECK_EQUAL(image.getColor(1, 0, c), 7);
        BOOST_CHECK_EQUAL(image.getColor(2, 0, c), 10);
        BOOST_CHECK_EQUAL(image.getColor(0, 1, c), 6);
        BOOST_CHECK_EQUAL(image.getColor(1, 1, c), 13);
        BOOST_CHECK_EQUAL(image.getColor(2, 1, c), 20);
        BOOST_CHECK_EQUAL(image.getColor(0, 2, c), 8);
        BOOST_CHECK_EQUAL(image.getColor(1, 2, c), 17);
        BOOST_CHECK_EQUAL(image.getColor(2, 2, c), 25);
    }

    image.calculateDerivative();

    for (int c = 0; c < 3; ++c) {
        BOOST_CHECK_EQUAL(image.getColor(0, 0, c), 5);
        BOOST_CHECK_EQUAL(image.getColor(1, 0, c), 2);
        BOOST_CHECK_EQUAL(image.getColor(2, 0, c), 3);
        BOOST_CHECK_EQUAL(image.getColor(0, 1, c), 1);
        BOOST_CHECK_EQUAL(image.getColor(1, 1, c), 5);
        BOOST_CHECK_EQUAL(image.getColor(2, 1, c), 4);
        BOOST_CHECK_EQUAL(image.getColor(0, 2, c), 2);
        BOOST_CHECK_EQUAL(image.getColor(1, 2, c), 2);
        BOOST_CHECK_EQUAL(image.getColor(2, 2, c), 1);
    }

}

BOOST_AUTO_TEST_CASE(testColorIntegralLargeImage) {
    RGBDImage image(640, 480);

    for (int y = 0; y < image.getHeight(); y++) {
        for (int x = 0; x < image.getWidth(); x++) {
            for (int c = 0; c < 3; c++) {
                image.setColor(x, y, c, 255.0f);
            }
        }
    }

    image.calculateIntegral();

    int width = 10;
    int height = 8;

    int x = image.getWidth() - width - 1;
    int y = image.getHeight() - height - 1;

    Point upperLeft(x - width, y - height);
    Point upperRight(x + width, y - height);
    Point lowerLeft(x - width, y + height);
    Point lowerRight(x + width, y + height);

    for (int c = 0; c < 3; c++) {
        double lowerRightPixel = image.getColor(lowerRight.getX(), lowerRight.getY(), c);
        double upperRightPixel = image.getColor(upperRight.getX(), upperRight.getY(), c);
        double lowerLeftPixel = image.getColor(lowerLeft.getX(), lowerLeft.getY(), c);
        double upperLeftPixel = image.getColor(upperLeft.getX(), upperLeft.getY(), c);

        double regionSum = lowerRightPixel - upperRightPixel - lowerLeftPixel + upperLeftPixel;
        BOOST_CHECK_CLOSE(regionSum, 2 * width * 2 * height * 255.0f, 0.0f);
    }
}

BOOST_AUTO_TEST_CASE(testColorIntegralLargeRandomImage) {

    const int SEED = 4711;
    Sampler sampler(SEED, 0, 255);

    RGBDImage randomImage(640, 480);
    for (int y = 0; y < randomImage.getHeight(); y++) {
        for (int x = 0; x < randomImage.getWidth(); x++) {
            for (int c = 0; c < 3; c++) {
                randomImage.setColor(x, y, c, sampler.getNext());
            }
        }
    }

    RGBDImage randomImageIntegrated(randomImage);

    randomImageIntegrated.calculateIntegral();

    int width = 5;
    int height = 4;

    int x = randomImageIntegrated.getWidth() - width - 1;
    int y = randomImageIntegrated.getHeight() - height - 1;

    Point upperLeft(x - width, y - height);
    Point upperRight(x + width, y - height);
    Point lowerLeft(x - width, y + height);
    Point lowerRight(x + width, y + height);

    for (int c = 0; c < 3; c++) {
        double lowerRightPixel = randomImageIntegrated.getColor(lowerRight.getX(), lowerRight.getY(), c);
        double upperRightPixel = randomImageIntegrated.getColor(upperRight.getX(), upperRight.getY(), c);
        double lowerLeftPixel = randomImageIntegrated.getColor(lowerLeft.getX(), lowerLeft.getY(), c);
        double upperLeftPixel = randomImageIntegrated.getColor(upperLeft.getX(), upperLeft.getY(), c);

        double regionSum = lowerRightPixel - upperRightPixel - lowerLeftPixel + upperLeftPixel;

        double expectedSum = 0.0;
        for (int y = upperLeft.getY() + 1; y <= lowerRight.getY(); y++) {
            for (int x = upperLeft.getX() + 1; x <= lowerRight.getX(); x++) {
                expectedSum += randomImage.getColor(x, y, c);
            }
        }

        CURFIL_DEBUG("actual sum: " << regionSum);
        CURFIL_DEBUG("expected sum: " << expectedSum);

        BOOST_CHECK_CLOSE(regionSum, expectedSum, 0.05f);
    }
}

BOOST_AUTO_TEST_CASE(testDepthIntegral) {

    std::cout << std::endl;
    RGBDImage image(3, 2);

    for (int y = 0; y < image.getHeight(); ++y) {
        for (int x = 0; x < image.getWidth(); ++x) {
            image.setDepth(x, y, Depth::INVALID);
        }
    }

    image.setDepth(0, 0, Depth(05.0f));
    image.setDepth(0, 1, Depth(15.0f));
    image.setDepth(0, 1, Depth(10.0f));
    image.setDepth(1, 0, Depth(35.0f));
    image.setDepth(2, 1, Depth(10.0f));

    image.dumpDepth(std::cout);
    std::cout << "valid" << std::endl;
    image.dumpDepthValid(std::cout);
    image.calculateIntegral();
    std::cout << "INTEGRAL" << std::endl;
    image.dumpDepth(std::cout);
    std::cout << "valid" << std::endl;
    image.dumpDepthValid(std::cout);

    /*
     * INTEGRAL
     *   0.5   4   4
     *   1.5   5   6
     * VALID
     *     1   2   2
     *     2   3   4
     */
    for (int y = 0; y < image.getHeight(); ++y) {
        for (int x = 0; x < image.getWidth(); ++x) {
            // depth integral must not contain NaNs
            BOOST_CHECK(image.getDepth(x, y).isValid());
            int nans = image.getDepthValid(x, y);
            BOOST_CHECK_GE(nans, 0);
        }
    }
    auto totalNans = image.getDepthValid(image.getWidth() - 1, image.getHeight() - 1);
    BOOST_CHECK_LT(totalNans, image.getWidth() * image.getHeight());

    BOOST_CHECK_EQUAL(image.getDepth(0, 0).getFloatValue(), 5);
    BOOST_CHECK_EQUAL(image.getDepth(1, 0).getFloatValue(), 40);
    BOOST_CHECK_EQUAL(image.getDepth(2, 0).getFloatValue(), 40);
    BOOST_CHECK_EQUAL(image.getDepth(0, 1).getFloatValue(), 15);
    BOOST_CHECK_EQUAL(image.getDepth(1, 1).getFloatValue(), 50);
    BOOST_CHECK_EQUAL(image.getDepth(2, 1).getFloatValue(), 60);

    BOOST_CHECK_EQUAL(image.getDepthValid(0, 0), 1);
    BOOST_CHECK_EQUAL(image.getDepthValid(1, 0), 2);
    BOOST_CHECK_EQUAL(image.getDepthValid(2, 0), 2);
    BOOST_CHECK_EQUAL(image.getDepthValid(0, 1), 2);
    BOOST_CHECK_EQUAL(image.getDepthValid(1, 1), 3);
    BOOST_CHECK_EQUAL(image.getDepthValid(2, 1), 4);
}

BOOST_AUTO_TEST_CASE(testWriteReadRGBDImage) {
    RGBDImage image(200, 100);

    const int DEPTH_PRECISION = 1000;
    const int COLOR_PRECISION = 100000;
    Sampler depthSampler(4711, 0, DEPTH_PRECISION);
    Sampler colorSampler(4711, 0, COLOR_PRECISION);

    for (int y = 0; y < image.getHeight(); y++) {
        for (int x = 0; x < image.getWidth(); x++) {
            float depth = 10.0f * (depthSampler.getNext() / static_cast<float>(DEPTH_PRECISION));
            if (depth < 1e-3) {
                depth = std::numeric_limits<double>::quiet_NaN();
            }
            image.setDepth(x, y, Depth(depth));
            for (unsigned int c = 0; c < 3; c++) {
                float color = colorSampler.getNext() / static_cast<float>(COLOR_PRECISION);
                assert(color >= 0.0 && color <= 1.0);
                image.setColor(x, y, c, color);
            }
        }
    }

    BOOST_REQUIRE(!fs::exists(temporaryColorFile));
    image.saveColor(temporaryColorFile.native());
    BOOST_REQUIRE(fs::exists(temporaryColorFile));

    BOOST_REQUIRE(!fs::exists(temporaryDepthFile));
    image.saveDepth(temporaryDepthFile.native());
    BOOST_REQUIRE(fs::exists(temporaryDepthFile));

    RGBDImage readImage(temporaryColorFile.native(), temporaryDepthFile.native(), true, false, false, false);

    for (int y = 0; y < image.getHeight(); y++) {
        for (int x = 0; x < image.getWidth(); x++) {
            const Depth expectedDepth = image.getDepth(x, y);
            const Depth actualDepth = readImage.getDepth(x, y);

            BOOST_REQUIRE_EQUAL(actualDepth.isValid(), expectedDepth.isValid());
            if (actualDepth.isValid()) {
                BOOST_REQUIRE_CLOSE_FRACTION(actualDepth.getFloatValue(), expectedDepth.getFloatValue(), 0.0);
            }

            for (unsigned int c = 0; c < 3; c++) {
                unsigned int expectedColor = image.getColor(x, y, c) * 0xFFFFu;
                unsigned int actualColor = readImage.getColor(x, y, c);
                BOOST_REQUIRE(abs(actualColor - expectedColor) <= 1);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testWriteIntegratedRGBDImage) {

    RGBDImage image(200, 100);

    const int COLOR_MAX = 64;
    Sampler colorSampler(4711, 0, COLOR_MAX);

    for (int y = 0; y < image.getHeight(); y++) {
        for (int x = 0; x < image.getWidth(); x++) {
            image.setDepth(x, y, Depth(1.0));
            for (unsigned int c = 0; c < 3; c++) {
                float color = colorSampler.getNext() / static_cast<float>(COLOR_MAX);
                image.setColor(x, y, c, color);
            }
        }
    }

    image.calculateIntegral();

    BOOST_REQUIRE(!fs::exists(temporaryColorFile));
    image.saveColor(temporaryColorFile.native());
    BOOST_REQUIRE(fs::exists(temporaryColorFile));

    BOOST_REQUIRE(!fs::exists(temporaryDepthFile));
    image.saveDepth(temporaryDepthFile.native());
    BOOST_REQUIRE(fs::exists(temporaryDepthFile));

    RGBDImage readImage(temporaryColorFile.native(), temporaryDepthFile.native(), true, false, false, false);

    image.calculateDerivative();

    for (int y = 0; y < image.getHeight(); y++) {
        for (int x = 0; x < image.getWidth(); x++) {
            const Depth expectedDepth = image.getDepth(x, y);
            const Depth actualDepth = readImage.getDepth(x, y);

            BOOST_REQUIRE_EQUAL(actualDepth.isValid(), expectedDepth.isValid());
            if (actualDepth.isValid()) {
                BOOST_REQUIRE_CLOSE_FRACTION(actualDepth.getFloatValue(), expectedDepth.getFloatValue(), 0.0);
            }

            for (unsigned int c = 0; c < 3; c++) {
                unsigned int expectedColor = image.getColor(x, y, c) * 0xFFFFu;
                unsigned int actualColor = readImage.getColor(x, y, c);
                BOOST_REQUIRE_EQUAL(actualColor, expectedColor);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testWriteReadLabelImage) {
    LabelImage image(200, 100);

    RGBColor color(0, 0, 0);
    addColorId(color, static_cast<LabelType>(0));

    for (int y = 10; y < 20; y++) {
        for (int x = 60; x < 80; x++) {
            color[0] = x;
            color[1] = y;
            LabelType label = (x + y) % std::numeric_limits<LabelType>::max();
            getOrAddColorId(color, label);
            image.setLabel(x, y, label);
            BOOST_REQUIRE_EQUAL(image.getLabel(x, y), label);
        }
    }

    image.save(temporaryColorFile.native());

    LabelImage readImage(temporaryColorFile.native());

    BOOST_REQUIRE_EQUAL(readImage.getLabel(0, 0), static_cast<LabelType>(0));

    for (int y = 10; y < 20; y++) {
        for (int x = 60; x < 80; x++) {
            LabelType label = (x + y) % std::numeric_limits<LabelType>::max();
            BOOST_REQUIRE_EQUAL(readImage.getLabel(x, y), label);
        }
    }

}

BOOST_AUTO_TEST_CASE(testLabeledRGBDImage) {

    boost::shared_ptr<RGBDImage> rgbdImage = boost::make_shared<RGBDImage>(300, 200);
    boost::shared_ptr<LabelImage> labelImage = boost::make_shared<LabelImage>(300, 200);

    LabeledRGBDImage labeledImage(rgbdImage, labelImage);

    BOOST_CHECK_THROW(LabeledRGBDImage(rgbdImage, boost::make_shared<LabelImage>(200, 300)), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(testDepthFilling) {

    RGBDImage image(200, 100);
    for (int y = 0; y < image.getHeight(); y++) {
        for (int x = 0; x < image.getWidth(); x++) {
            image.setDepth(x, y, Depth::INVALID);
        }
    }

    image.setDepth(20, 30, Depth(1.5));
    image.setDepth(30, 20, Depth(2.5));

    BOOST_REQUIRE_EQUAL(image.getDepthValid(0, 0), 0);
    BOOST_REQUIRE_EQUAL(image.getDepthValid(20, 30), 1);
    BOOST_REQUIRE_EQUAL(image.getDepthValid(30, 20), 1);

    image.fillDepth();

    for (int y = 0; y < image.getHeight(); y++) {
        for (int x = 0; x < image.getWidth(); x++) {
            const Depth& depth = image.getDepth(x, y);
            BOOST_REQUIRE(depth.isValid());

            if (y < 30) {
                BOOST_REQUIRE_EQUAL(2.5, depth.getFloatValue());
            } else {
                BOOST_REQUIRE_EQUAL(1.5, depth.getFloatValue());
            }
        }
    }

}
BOOST_AUTO_TEST_SUITE_END()
