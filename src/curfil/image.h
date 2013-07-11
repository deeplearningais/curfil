#ifndef CURFIL_IMAGE_H
#define CURFIL_IMAGE_H

#include <boost/shared_ptr.hpp>
#include <cassert>
#include <cuv/ndarray.hpp>
#include <limits>
#include <memory>
#include <ostream>
#include <stdint.h>
#include <string>
#include <vector>

#include "utils.h"

namespace curfil {

typedef uint8_t LabelType;

class RGBColor: public std::vector<uint8_t> {
public:

    RGBColor();

    RGBColor(uint8_t r, uint8_t g, uint8_t b);

    RGBColor(const std::string& colorString);

    std::string toString() const {
        std::ostringstream o;
        o << static_cast<int>((*this)[0]) << ",";
        o << static_cast<int>((*this)[1]) << ",";
        o << static_cast<int>((*this)[2]);
        return o.str();
    }

    friend std::ostream& operator<<(std::ostream& os, RGBColor const& color) {
        return os << color.toString();
    }

};

LabelType getOrAddColorId(const RGBColor& color, const LabelType& label);

void addColorId(const RGBColor& color, const LabelType& label);

class Depth
{
public:
    Depth() :
            value(0) {
    }

    explicit Depth(const int& value) :
            value(value) {
    }

    explicit Depth(const double& value) :
            value(1000.0 * value) {
        if (value != value) { // nan
            this->value = -1;
            assert(!isValid());
        } else {
            assert(value >= -1);
        }
    }

    bool isValid() const {
        return (value > 0);
    }

    float getFloatValue() const {
        assert(isValid());
        return value / 1000.0f;
    }

    int getIntValue() const {
        return value;
    }

    Depth operator-(const Depth& other) const {
        assert(other.getIntValue() >= 0);
        return Depth(value - other.value);
    }

    Depth operator+(const Depth& other) const {
        assert(other.getIntValue() >= 0);
        return Depth(value + other.value);
    }

    Depth& operator+=(const Depth& other) {
        assert(other.getIntValue() >= 0);
        value += other.value;
        return (*this);
    }

    static const Depth INVALID;

private:

    int value;
};

class RGBDImage {

private:

    std::string filename;
    std::string depthFilename;
    int width;
    int height;
    cuv::ndarray<float, cuv::host_memory_space> colorImage;
    cuv::ndarray<int, cuv::host_memory_space> depthImage;

    bool inCIELab;
    bool integratedColor;
    bool integratedDepth;

    static const unsigned int COLOR_CHANNELS = 3;
    static const unsigned int DEPTH_CHANNELS = 2;

    void loadDepthImage(const std::string& depthFilename);

    void fillDepthFromRight();
    void fillDepthFromLeft();
    void fillDepthFromBottom();
    void fillDepthFromTop();

    template<class T>
    static void calculateDerivative(cuv::ndarray_view<T, cuv::host_memory_space>& data);

    template<class T>
    static void calculateIntegral(cuv::ndarray_view<T, cuv::host_memory_space>& data);

    static void calculateIntegral(cuv::ndarray_view<float, cuv::host_memory_space>& data);
    static void calculateDerivative(cuv::ndarray_view<float, cuv::host_memory_space>& data);

public:

    explicit RGBDImage(const std::string& filename, const std::string& depthFilename,
            bool convertToCIELab = true,
            bool useDepthFilling = false,
            bool calculateIntegralImage = true);

    // for the test case
    explicit RGBDImage(int width, int height) :
            filename(""), depthFilename(""),
                    width(width), height(height),
                    colorImage(cuv::extents[COLOR_CHANNELS][height][width], boost::make_shared<cuv::cuda_allocator>()),
                    depthImage(cuv::extents[DEPTH_CHANNELS][height][width], boost::make_shared<cuv::cuda_allocator>()),
                    inCIELab(false), integratedColor(false), integratedDepth(false) {
        assert(width >= 0 && height >= 0);
        reset();
    }

    RGBDImage(const RGBDImage& other);

    size_t getSizeInMemory() const {
        return colorImage.size() * sizeof(float) + depthImage.size() * sizeof(int);
    }

    /**
     * simple depth filling
     * depth values are replaced by right,left,top,bottom neighbors (in that order)
     */
    void fillDepth();

    void reset();

    const cuv::ndarray<float, cuv::host_memory_space>& getColorImage() const {
        return colorImage;
    }

    const cuv::ndarray<int, cuv::host_memory_space>& getDepthImage() const {
        return depthImage;
    }

    // public for test case
    void calculateDerivative();
    void calculateIntegral();

    void dump(std::ostream& out) const;
    void dumpDepth(std::ostream& out) const;
    void dumpDepthValid(std::ostream& out) const;

    void saveColor(const std::string& filename) const;
    void saveDepth(const std::string& filename) const;

    // GETTERS / SETTERS
    const std::string& getFilename() const {
        return filename;
    }

    int getWidth() const {
        return width;
    }

    int getHeight() const {
        return height;
    }

    bool hasIntegratedDepth() const {
        return integratedDepth;
    }

    bool hasIntegratedColor() const {
        return integratedColor;
    }

    bool inImage(int x, int y) const {
        if (x < 0 || x >= getWidth()) {
            return false;
        }
        if (y < 0 || y >= getHeight()) {
            return false;
        }
        return true;
    }

    void setDepth(int x, int y, const Depth& depth) {
        assert(inImage(x, y));

        // invalid depth is set to zero which is necessary for integrating
        depthImage(0, y, x) = depth.isValid() ? depth.getIntValue() : 0;
        depthImage(1, y, x) = depth.isValid();
    }

    Depth getDepth(int x, int y) const {
        return Depth(static_cast<int>(depthImage(0, y, x)));
    }

    int getDepthValid(int x, int y) const {
        return depthImage(1, y, x);
    }

    void setColor(int x, int y, unsigned int channel, float color) {
        // colorImage(channel, y, x) = color;
        colorImage.ptr()[channel * getWidth() * getHeight() + y * getWidth() + x] = color;
    }

    float getColor(int x, int y, unsigned int channel) const {
        // return colorImage(channel, y, x);
        return colorImage.ptr()[channel * getWidth() * getHeight() + y * getWidth() + x];
    }

};

class LabelImage {

private:

    std::string filename;
    int width;
    int height;
    cuv::ndarray<LabelType, cuv::host_memory_space> image;

public:

    LabelImage(int width, int height) :
            width(width), height(height), image(height, width) {
        image = LabelType();
        assert(width >= 0 && height >= 0);
#ifndef NDEBUG
        if (width > 0 && height > 0) {
            assert(image(0, 0) == static_cast<LabelType>(0));
        }
#endif
    }

    LabelImage(const std::string& filename);

    const std::string& getFilename() const {
        return filename;
    }

    bool isInImage(int x, int y) const {
        if (x < 0 || x >= width)
            return false;
        if (y < 0 || y >= height)
            return false;

        return true;
    }

    void save(const std::string& filename) const;

    static RGBColor decodeLabel(const LabelType& v);

    size_t getSizeInMemory() const {
        return image.size() * sizeof(LabelType);
    }

    int getWidth() const {
        return width;
    }

    int getHeight() const {
        return height;
    }

    void setLabel(int x, int y, const LabelType label) {
        assert(isInImage(x, y));
        image(y, x) = label;
    }

    LabelType getLabel(int x, int y) const {
        assert(isInImage(x, y));
        return image(y, x);
    }

}
;

class LabeledRGBDImage {

public:
    boost::shared_ptr<RGBDImage> rgbdImage;
    boost::shared_ptr<LabelImage> labelImage;

public:

    LabeledRGBDImage() :
            rgbdImage(), labelImage() {
    }

    LabeledRGBDImage(const boost::shared_ptr<RGBDImage>& rgbdImage,
            const boost::shared_ptr<LabelImage>& labelImage);

    size_t getSizeInMemory() const {
        return rgbdImage->getSizeInMemory() + labelImage->getSizeInMemory();
    }

    const RGBDImage& getRGBDImage() const {
        return *(rgbdImage.get());
    }

    const LabelImage& getLabelImage() const {
        return *(labelImage.get());
    }

    int getWidth() const {
        return rgbdImage->getWidth();
    }

    int getHeight() const {
        return rgbdImage->getHeight();
    }

};

LabeledRGBDImage loadImagePair(const std::string& filename, bool useCIELab, bool useDepthFilling,
        bool calculateIntegralImages = true);

std::vector<std::string> listImageFilenames(const std::string& path);

std::vector<LabeledRGBDImage> loadImages(const std::string& folder, bool useCIELab, bool useDepthFilling);

}

#endif
