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

/**
 * A tuple of three 8-bit uints that represent a RGB color value
 * @ingroup image_related
 */
class RGBColor: public std::vector<uint8_t> {

public:

    RGBColor();

    /**
     * Creates an RGB color from red, green and blue values
     */
    RGBColor(uint8_t r, uint8_t g, uint8_t b);

    /**
     * Creates a RGB color from a string.
     *
     * Example:
     *   RGBColor("255,128,0") == RGBColor(0x255, 0x128, 0x0)
     */
    RGBColor(const std::string& colorString);

    /**
     * convert the RGB color to string.
     *
     * Example:
     *  RGBColor(0x255, 0x128, 0x0).toString() == "255,128,0"
     */
    std::string toString() const {
        std::ostringstream o;
        o << static_cast<int>((*this)[0]) << ",";
        o << static_cast<int>((*this)[1]) << ",";
        o << static_cast<int>((*this)[2]);
        return o.str();
    }

    /**
     * add the string representation of the color to the stream
     */
    friend std::ostream& operator<<(std::ostream& os, RGBColor const& color) {
        return os << color.toString();
    }

};

LabelType getOrAddColorId(const RGBColor& color, const LabelType& label);

void addColorId(const RGBColor& color, const LabelType& label);

/**
 * Wrapper class that represent a depth value as it occurs in RGB-D images.
 * @ingroup image_related
 *
 * The internal representation is an int that represent the depth (distance) in millimeter precision.
 *
 * A depth of zero represents an invalid depth that can be used to represent a missing depth value
 * as it occurs in RGB-D cameras such as the Microsoft Kinect.
 */
class Depth
{
public:
    Depth() :
            value(0) {
    }

    /**
     * @param value the depth (distance) in millimeter
     */
    explicit Depth(const int& value) :
            value(value) {
    }

    /**
     * @param value the depth (distance) in meter
     */
    explicit Depth(const double& value) :
            value(1000.0 * value) {
        if (value != value) { // nan
            this->value = -1;
            assert(!isValid());
        } else {
            assert(value >= -1);
        }
    }

    /**
     * @return true if and only if the depth is strictly greater than zero and represents a valid depth measure
     */
    bool isValid() const {
        return (value > 0);
    }

    /**
     * @return the depth in meter as a float.
     */
    float getFloatValue() const {
        assert(isValid());
        return value / 1000.0f;
    }

    /**
     * @return the depth in millimeter. can be zero which indicates an invalid depth
     */
    int getIntValue() const {
        return value;
    }

    /**
     * @return the difference in depth
     */
    Depth operator-(const Depth& other) const {
        assert(other.getIntValue() >= 0);
        return Depth(value - other.value);
    }

    /**
     * @return the combined depth values
     */
    Depth operator+(const Depth& other) const {
        assert(other.getIntValue() >= 0);
        return Depth(value + other.value);
    }

    /**
     * @return the depth after adding to it another value
     */
    Depth& operator+=(const Depth& other) {
        assert(other.getIntValue() >= 0);
        value += other.value;
        return (*this);
    }

    /**
     * Static value used when the depth is invalid, e.gs outside the image.
     */
    static const Depth INVALID;

private:

    int value;
};

/**
 * An RGB-D image that contains four channels for the RGB color and the depth.
 * @ingroup image_related
 *
 * The class provides convenience methods to convert the image between RGB and CIELab color space
 * and to calculate image integrals.
 *
 * Image loading and saving is implemented using the vigraimpex library.
 *
 * The image is stored as compact matrix in row-major order using cuv::ndarray.
 */
class RGBDImage {


public:

	/**
	 * Load the RGB color image and the according depth from two files on disk.
	 * See the README file for the required file format.
	 */

    explicit RGBDImage(const std::string& filename, const std::string& depthFilename, bool useDepthImages,
            bool convertToCIELab = true,
            bool useDepthFilling = false,
            bool calculateIntegralImage = true);

    /**
     * For the test case
     */
    explicit RGBDImage(int width, int height) :
            filename(""), depthFilename(""),
                    width(width), height(height),
                    colorImage(cuv::extents[COLOR_CHANNELS][height][width], boost::make_shared<cuv::cuda_allocator>()),
                    depthImage(cuv::extents[DEPTH_CHANNELS][height][width], boost::make_shared<cuv::cuda_allocator>()),
                    inCIELab(false), integratedColor(false), integratedDepth(false) {
        assert(width >= 0 && height >= 0);
        reset();
    }

    /**
     * uses the attributes of another object to set its attributes
     */
    RGBDImage(const RGBDImage& other);

    /**
     * @return total size in memory in bytes
     */
    size_t getSizeInMemory() const {
        return colorImage.size() * sizeof(float) + depthImage.size() * sizeof(int);
    }

    /**
     * Simple depth filling
     *
     * Depth values are replaced by right,left,top,bottom neighbors (in that order)
     */
    void fillDepth();

    /**
     * resets all values to zero
     */
    void reset();

    /**
     * @return a constant reference on the underlying RGB color matrix
     */
    const cuv::ndarray<float, cuv::host_memory_space>& getColorImage() const {
        return colorImage;
    }

    /**
     * @return a constant reference on the underlying depth matrix
     */
    const cuv::ndarray<int, cuv::host_memory_space>& getDepthImage() const {
        return depthImage;
    }

    /**
     * Calculate the image derivative
     *
     * This method is only allowed if the image is already integrated.
     */
    void calculateDerivative();

    /**
     * Calculate the image integral
     *
     * This method is only allowed if the image is not already integrated.
     */
    void calculateIntegral();

    /**
     * Print the image color channels in a human-readable format to the output stream.
     *
     * Used for debugging purposes and the test case.
     */
    void dump(std::ostream& out) const;

    /**
     * Print the image depth channel in a human-readable format to the output stream.
     *
     * Used for debugging purposes and the test case.
     */
    void dumpDepth(std::ostream& out) const;

    /**
     * Print the image depth valid channel in a human-readable format to the output stream.
     *
     * Used for debugging purposes and the test case.
     */
    void dumpDepthValid(std::ostream& out) const;

    /**
     * save (export) the image color channels to a file
     */
    void saveColor(const std::string& filename) const;

    /**
     * save (export) the image depth channels to a file
     */
    void saveDepth(const std::string& filename) const;

    /**
     * @return the name of the file this image was loaded from
     */
    const std::string& getFilename() const {
        return filename;
    }

    /**
     * @return width of the image in pixels
     */
    int getWidth() const {
        return width;
    }

    /**
     * @return height of the image in pixels
     */
    int getHeight() const {
        return height;
    }

    /**
     * @return true if and only if the image depth channel was integrated
     */
    bool hasIntegratedDepth() const {
        return integratedDepth;
    }

    /**
     * @return true if and only if the image color channel was integrated
     */
    bool hasIntegratedColor() const {
        return integratedColor;
    }

    /**
     * @return true if and only if the specified position lies in the image
     */
    bool inImage(int x, int y) const {
        if (x < 0 || x >= getWidth()) {
            return false;
        }
        if (y < 0 || y >= getHeight()) {
            return false;
        }
        return true;
    }

    /**
     * Sets a new depth at the given position. Be careful when the image is already integrated!
     */
    void setDepth(int x, int y, const Depth& depth) {
        assert(inImage(x, y));

        // invalid depth is set to zero which is necessary for integrating
        depthImage(0, y, x) = depth.isValid() ? depth.getIntValue() : 0;
        depthImage(1, y, x) = depth.isValid();
    }

    /**
     * @return the depth value at the given position. Note: Can be an integrated value!
     */
    Depth getDepth(int x, int y) const {
        return Depth(static_cast<int>(depthImage(0, y, x)));
    }

    /**
     * @return the depth validity information. 0 or 1 if the image is not integrated.
     */
    int getDepthValid(int x, int y) const {
        return depthImage(1, y, x);
    }

    /**
     * sets a new color value at the given position and color channel
     *
     * @param x the x position in the image where 0 <= x < width
     * @param y the y position in the image where 0 <= y < height
     * @param channel the color channel where 0 <= channel < COLOR_CHANNELS
     * @param color the new color value
     */
    void setColor(int x, int y, unsigned int channel, float color) {
        // colorImage(channel, y, x) = color;
        colorImage.ptr()[channel * getWidth() * getHeight() + y * getWidth() + x] = color;
    }

    /**
     * @param x the x position in the image where 0 <= x < width
     * @param y the y position in the image where 0 <= y < height
     * @param channel the color channel where 0 <= channel < COLOR_CHANNELS
     * @return the color value at the given position and color channel
     */
    float getColor(int x, int y, unsigned int channel) const {
        // return colorImage(channel, y, x);
        return colorImage.ptr()[channel * getWidth() * getHeight() + y * getWidth() + x];
    }

    /**
     * resize an image to a new depth and height
     */
    void resizeImage(int newWidth, int newHeight);

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
    void loadDummyDepthValues();
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

};

/**
 * A labelling that usually belongs to a RGBImage.
 * @ingroup image_related
 *
 * @see LabeledRGBDImage
 */
class LabelImage {

private:

    std::string filename;
    int width;
    int height;
    cuv::ndarray<LabelType, cuv::host_memory_space> image;

public:

    /**
     * Creates a label image of the given width and height.
     * Initial label values are set to zero.
     */
    LabelImage(int width, int height) :
            filename(), width(width), height(height), image(height, width) {
        image = LabelType();
        assert(width >= 0 && height >= 0);
#ifndef NDEBUG
        if (width > 0 && height > 0) {
            assert(image(0, 0) == static_cast<LabelType>(0));
        }
#endif
    }

    /**
     * Loads a label image from an image file from disk. Each color in the image is assigned to a unique color id.
     */
    LabelImage(const std::string& filename);

    /**
     * @return the filename this label image was loaded from. Empty if it was created manually.
     */
    const std::string& getFilename() const {
        return filename;
    }

    /**
     * @return true if and only if the position lies in the image
     */
    bool isInImage(int x, int y) const {
        if (x < 0 || x >= width)
            return false;
        if (y < 0 || y >= height)
            return false;

        return true;
    }

    /**
     * store (export) this label image to the given filename
     */
    void save(const std::string& filename) const;

    /**
     * convert the internal unique label id to a RGBColor
     */
    static RGBColor decodeLabel(const LabelType& v);

    /**
     * @return the total memory usage of this image in bytes
     */
    size_t getSizeInMemory() const {
        return image.size() * sizeof(LabelType);
    }

    /**
     * @return the width of the image
     */
    int getWidth() const {
        return width;
    }

    /**
     * @return the height of the image
     */
    int getHeight() const {
        return height;
    }

    /**
     * sets the label of a pixel
     */
    void setLabel(int x, int y, const LabelType label) {
        assert(isInImage(x, y));
        image(y, x) = label;
    }

    /**
     * @return the label of a pixel
     */
    LabelType getLabel(int x, int y) const {
        assert(isInImage(x, y));
        return image(y, x);
    }

    /**
     * @return the id associated with the color
     */
    static LabelType encodeColor(RGBColor color);

    /**
     * resize the image to new depth and height and uses the provided label for padding
     */
    void resizeImage(int newWidth, int newHeight, LabelType paddingLabel);

}
;

/**
 * A tuple of a RGBD image and an according labeling.
 * @ingroup image_related
 */
class LabeledRGBDImage {

public:
	/**
	 * the RGBD component
	 */
    boost::shared_ptr<RGBDImage> rgbdImage;
    /**
     * the label iamge component
     */
    boost::shared_ptr<LabelImage> labelImage;

public:

    LabeledRGBDImage() :
            rgbdImage(), labelImage() {
    }

    /**
     * creates a LabeledRGBDImage object out of an rgbd and label images
     */
    LabeledRGBDImage(const boost::shared_ptr<RGBDImage>& rgbdImage,
            const boost::shared_ptr<LabelImage>& labelImage);

    /**
     * @return the total memory usage of the RGBD image as well as the according label image
     */
    size_t getSizeInMemory() const {
        return rgbdImage->getSizeInMemory() + labelImage->getSizeInMemory();
    }

    /**
     * @return the RGBD image component
     */
    const RGBDImage& getRGBDImage() const {
        return *(rgbdImage.get());
    }

    /**
     * @return the label image component
     */
    const LabelImage& getLabelImage() const {
        return *(labelImage.get());
    }

    /**
     * @return the width of the RGBD image
     */
    int getWidth() const {
        return rgbdImage->getWidth();
    }

    /**
     * @return the height of the RGBD image
     */
    int getHeight() const {
        return rgbdImage->getHeight();
    }

    /**
     * resize the image to new depth and height and uses the provided label for padding
     */
    void resizeImage(int newWidth, int newHeight, LabelType paddingLabel) const;

    /**
     * calculate the image integral
     */
    void calculateIntegral() const;

};


/**
 * Convenience function to load and convert a RGBD image and the according label image
 */
LabeledRGBDImage loadImagePair(const std::string& filename, bool useCIELab, bool useDepthImages,  bool useDepthFilling,
        bool calculateIntegralImages = true);

/**
 * Convenience function to find all files in the given folder that match the required filename schema.
 * See the README for the filename schema.
 */
std::vector<std::string> listImageFilenames(const std::string& folder);

/**
 * Convenience function to find all files in the given folder that match the required filename schema.
 * See the README for the filename schema.
 */
std::vector<LabeledRGBDImage> loadImages(const std::string& folder, bool useCIELab, bool useDepthImages, bool useDepthFilling, const std::vector<std::string>& ignoredColors, size_t& numLabels);

}

#endif
