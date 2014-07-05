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
#ifndef CURFIL_RANDOMTREEIMAGE_H
#define CURFIL_RANDOMTREEIMAGE_H

#include <algorithm>
#include <assert.h>
#include <boost/make_shared.hpp>
#include <cuv/ndarray.hpp>
#include <list>
#include <stdint.h>
#include <vector>

#include "image.h"
#include "random_tree.h"

namespace curfil {

/**
 * A simple 2D tuple.
 * @ingroup import_export_trees
 *
 * Instances of this classes can be used as point (in two-dimensional) space or offset.
 *
 * Can be normalized (scaled) according to a given depth.
 */
class XY {

public:
    XY() :
            x(0), y(0) {
    }
    /**
     * create a 2D tuple using the given x, y
     */
    XY(int x, int y) :
            x(x), y(y) {
    }
    /**
     * create a 2D tuple using another tuple attributes
     */
    XY(const XY& other) :
            x(other.x), y(other.y) {
    }
    /**
     * set the attributes using another tuple 
     */
    XY& operator=(const XY& other) {
        x = other.x;
        y = other.y;
        return (*this);
    }

    /**
     * Normalize the x and y coordindate such that x ← x/depth and y ← y/depth
     */
    XY normalize(const Depth& depth) const {
        assert(depth.isValid());
        int newX = static_cast<int>(x / depth.getFloatValue());
        int newY = static_cast<int>(y / depth.getFloatValue());
        return XY(newX, newY);
    }

    /**
     * @return whether the tuple is equal to another
     */
    bool operator==(const XY& other) const {
        return (x == other.x && y == other.y);
    }

    /**
     * @return whether the tuple is not equal to another
     */
    bool operator!=(const XY& other) const {
        return !(*this == other);
    }

    /**
     * @return the x coordinate
     */
    int getX() const {
        return x;
    }

    /**
     * @return the y coordinate
     */
    int getY() const {
        return y;
    }

private:
    int x, y;
};

typedef XY Region;
typedef XY Offset;
typedef XY Point;

/**
 * Represents a single-pixel sample in a RGB-D image.
 * @ingroup forest_hierarchy
 */
class PixelInstance {

public:

    /**
     * @param image the RGB-D image that contains this pixel. Must not be null/
     * @param label the ground-truth labeleing of this pixel.
     * @param x the x-coordinate of the pixel in the RGB-D image
     * @param y the y-coordinate of the pixel in the RGB-D image
     * @param setting horizontal flip setting, can be: NoFlip, Flip, Both
     */
    PixelInstance(const RGBDImage* image, const LabelType& label, uint16_t x, uint16_t y, HorizontalFlipSetting setting = NoFlip) :
            image(image), label(label), point(x, y), depth(Depth::INVALID), horFlipSetting(setting) {
        assert(image != NULL);
        assert(image->inImage(x, y));
        if (!image->hasIntegratedDepth()) {
            throw std::runtime_error("image is not integrated");
        }

        int aboveValid = (y > 0) ? image->getDepthValid(x, y - 1) : 0;
        int leftValid = (x > 0) ? image->getDepthValid(x - 1, y) : 0;
        int aboveLeftValid = (x > 0 && y > 0) ? image->getDepthValid(x - 1, y - 1) : 0;

        int valid = image->getDepthValid(x, y) - (leftValid + aboveValid - aboveLeftValid);
        assert(valid == 0 || valid == 1);

        if (valid == 1) {
            Depth above = (y > 0) ? image->getDepth(x, y - 1) : Depth(0);
            Depth left = (x > 0) ? image->getDepth(x - 1, y) : Depth(0);
            Depth aboveLeft = (x > 0 && y > 0) ? image->getDepth(x - 1, y - 1) : Depth(0);

            depth = image->getDepth(x, y) - (left + above - aboveLeft);
            assert(depth.isValid());
        } else {
            assert(!depth.isValid());
        }
    }

    /**
     * @param image the RGB-D image that contains this pixel. Must not be null/
     * @param label the ground-truth labeleing of this pixel.
     * @param depth the depth of the pixel in the RGB-D image
     * @param x the x-coordinate of the pixel in the RGB-D image
     * @param y the y-coordinate of the pixel in the RGB-D image
     * @param setting horizontal flip setting, can be: NoFlip, Flip, Both
     */
    PixelInstance(const RGBDImage* image, const LabelType& label, const Depth& depth,
            uint16_t x, uint16_t y, HorizontalFlipSetting setting = NoFlip) :
            image(image), label(label), point(x, y), depth(depth), horFlipSetting(setting) {
        assert(image != NULL);
        assert(image->inImage(x, y));
        assert(depth.isValid());
    }

    /**
     * @return pointer to the RGB-D image which contains this pixel
     */
    const RGBDImage* getRGBDImage() const {
        return image;
    }

    /**
     * @return width in pixels of the RGB-D image which contains this pixel
     */
    int width() const {
        return image->getWidth();
    }

    /**
     * @return height in pixels of the RGB-D image which contains this pixel
     */
    int height() const {
        return image->getHeight();
    }

    /**
     * @return the x-coordinate of the pixel in the RGB-D image which contains this pixel
     */
    uint16_t getX() const {
        return static_cast<uint16_t>(point.getX());
    }

    /**
     * @return the x-coordinate of the pixel in the RGB-D image which contains this pixel
     */
    uint16_t getY() const {
        return static_cast<uint16_t>(point.getY());
    }

    /**
     * Calculate the average value for the given region at the offset and the given color channel, on CPU.
     * See https://github.com/deeplearningais/curfil/wiki/Visual-Features for details.
     */
    FeatureResponseType averageRegionColor(const Offset& offset, const Region& region, uint8_t channel) const {

        assert(region.getX() >= 0);
        assert(region.getY() >= 0);

        assert(image->hasIntegratedColor());

        const int width = std::max(1, region.getX());
        const int height = std::max(1, region.getY());

        int x = getX() + offset.getX();
        int y = getY() + offset.getY();

        int leftX = x - width;
        int rightX = x + width;
        int upperY = y - height;
        int lowerY = y + height;

        if (leftX < 0 || rightX >= image->getWidth() || upperY < 0 || lowerY >= image->getHeight()) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        assert(inImage(x, y));

        Point upperLeft(leftX, upperY);
        Point upperRight(rightX, upperY);
        Point lowerLeft(leftX, lowerY);
        Point lowerRight(rightX, lowerY);

        FeatureResponseType lowerRightPixel = getColor(lowerRight, channel);
        FeatureResponseType lowerLeftPixel = getColor(lowerLeft, channel);
        FeatureResponseType upperRightPixel = getColor(upperRight, channel);
        FeatureResponseType upperLeftPixel = getColor(upperLeft, channel);

        if (isnan(lowerRightPixel) || isnan(lowerLeftPixel) || isnan(upperRightPixel) || isnan(upperLeftPixel))
        	   return std::numeric_limits<double>::quiet_NaN();

        FeatureResponseType sum = (lowerRightPixel - upperRightPixel) + (upperLeftPixel - lowerLeftPixel);

        return sum;
    }

    /**
     * Calculate the average depth for the given region at the offset, on CPU.
     * See https://github.com/deeplearningais/curfil/wiki/Visual-Features for details.
     */
    FeatureResponseType averageRegionDepth(const Offset& offset, const Region& region) const {
        assert(region.getX() >= 0);
        assert(region.getY() >= 0);

        assert(image->hasIntegratedDepth());

        const int width = std::max(1, region.getX());
        const int height = std::max(1, region.getY());

        int x = getX() + offset.getX();
        int y = getY() + offset.getY();

        int leftX = x - width;
        int rightX = x + width;
        int upperY = y - height;
        int lowerY = y + height;

        if (leftX < 0 || rightX >= image->getWidth() || upperY < 0 || lowerY >= image->getHeight()) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        assert(inImage(x, y));

        Point upperLeft(leftX, upperY);
        Point upperRight(rightX, upperY);
        Point lowerLeft(leftX, lowerY);
        Point lowerRight(rightX, lowerY);

        int upperLeftValid = getDepthValid(upperLeft);
        int upperRightValid = getDepthValid(upperRight);
        int lowerRightValid = getDepthValid(lowerRight);
        int lowerLeftValid = getDepthValid(lowerLeft);

        int numValid = (lowerRightValid - upperRightValid) + (upperLeftValid - lowerLeftValid);
        assert(numValid >= 0);

        if (numValid == 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const int lowerRightDepth = getDepth(lowerRight).getIntValue();
        const int lowerLeftDepth = getDepth(lowerLeft).getIntValue();
        const int upperRightDepth = getDepth(upperRight).getIntValue();
        const int upperLeftDepth = getDepth(upperLeft).getIntValue();

        int sum = (lowerRightDepth - upperRightDepth) + (upperLeftDepth - lowerLeftDepth);
        FeatureResponseType feat = sum / static_cast<FeatureResponseType>(1000);
        return (feat / numValid);
    }

    /**
     * @return the ground-truth labeling of this pixel
     */
    LabelType getLabel() const {
        return label;
    }

    /**
     * @return the according depth of this pixel in the RGB-D image
     */
    const Depth getDepth() const {
        return depth;
    }

    /**
     * Currently not used and always returns 1.
     */
    WeightType getWeight() const {
        return 1;
    }

    /**
     * @return the horizontal flip setting
     */
    HorizontalFlipSetting getHorFlipSetting() const
    {
    	return horFlipSetting;
    }

    /**
     * use the new horizontal flip setting
     */
    void setHorFlipSetting(HorizontalFlipSetting setting)
    {
    	horFlipSetting = setting;
    }

private:
    const RGBDImage* image;
    LabelType label;
    Point point;
    Depth depth;
    HorizontalFlipSetting horFlipSetting;

    float getColor(const Point& pos, uint8_t channel) const {
        if (!inImage(pos)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        assert(image->hasIntegratedColor());
        return image->getColor(pos.getX(), pos.getY(), channel);
    }

    Depth getDepth(const Point& pos) const {
        if (!inImage(pos)) {
            return Depth::INVALID;
        }
        assert(image->hasIntegratedDepth());
        const Depth depth = image->getDepth(pos.getX(), pos.getY());
        // include zero as it is an integral
        assert(depth.getIntValue() >= 0);
        return depth;
    }

    int getDepthValid(const Point& pos) const {
        return image->getDepthValid(pos.getX(), pos.getY());
    }

    bool inImage(int x, int y) const {
        return image->inImage(x, y);
    }

    bool inImage(const Point& pos) const {
        return inImage(pos.getX(), pos.getY());
    }
};

enum FeatureType {
    DEPTH = 0, COLOR = 1
};

/**
 * Parametrized visual image feature.
 * @ingroup training_helper_classes
 *
 * See https://github.com/deeplearningais/curfil/wiki/Visual-Features for details.
 */
class ImageFeatureFunction {

public:

    /**
     * @param featureType either COLOR of DEPTH
     * @param offset1 the offset of the first region
     * @param region1 the extent size of the first region
     * @param channel1 the image channel that the first region belongs to. Only used for COLOR features.
     * @param offset2 the offset of the second region
     * @param region2 the extent size of the second region
     * @param channel2 the image channel that the second region belongs to. Only used for COLOR features.
     */
    ImageFeatureFunction(FeatureType featureType,
            const Offset& offset1,
            const Region& region1,
            const uint8_t channel1,
            const Offset& offset2,
            const Region& region2,
            const uint8_t channel2) :
            featureType(featureType),
                    offset1(offset1),
                    region1(region1),
                    channel1(channel1),
                    offset2(offset2),
                    region2(region2),
                    channel2(channel2) {
        if (offset1 == offset2) {
            throw std::runtime_error("illegal feature: offset1 equals offset2");
        }
        assert(isValid());
    }

    ImageFeatureFunction() :
            featureType(), offset1(), region1(), channel1(), offset2(), region2(), channel2() {
    }

    /**
     * @return a 32-bit key that is used to sort the features for performance reasons (improve cache-hit rate).
     */
    int getSortKey() const {
        int32_t sortKey = 0;
        sortKey |= static_cast<uint8_t>(getType() & 0x03) << 30; // 2 bit for the type
        sortKey |= static_cast<uint8_t>(getChannel1() & 0x0F) << 26; // 4 bit for channel1
        sortKey |= static_cast<uint8_t>(getChannel2() & 0x0F) << 22; // 4 bit for channel2
        sortKey |= static_cast<uint8_t>((getOffset1().getY() + 127) & 0xFF) << 14; // 8 bit for offset1.y
        sortKey |= static_cast<uint8_t>((getOffset1().getX() + 127) & 0xFF) << 6; // 8 bit for offset1.x
        return sortKey;
    }

    /**
     * @return the feature type. e.g. COLOR or DEPTH
     */
    FeatureType getType() const {
        return featureType;
    }

    /**
     * @return the feature type as string in lowercase. e.g. "color" or "depth".
     */
    std::string getTypeString() const {
        switch (featureType) {
            case COLOR:
                return "color";
            case DEPTH:
                return "depth";
            default:
                throw std::runtime_error("unknown feature");
        }
    }

    /**
     * @return true if and only if the parameters of this feature are valid
     */
    bool isValid() const {
        return (offset1 != offset2);
    }

    /**
     * @return the feature response as documented on https://github.com/deeplearningais/curfil/wiki/Visual-Features.
     */
    FeatureResponseType calculateFeatureResponse(const PixelInstance& instance, bool flipRegion = false) const {
        assert(isValid());
        switch (featureType) {
            case DEPTH:
                return calculateDepthFeature(instance, flipRegion);
            case COLOR:
                return calculateColorFeature(instance, flipRegion);
            default:
                assert(false);
                break;
        }
        return 0;
    }

    /**
     * @return the x/y offset in pixels of the first region.
     */
    const Offset& getOffset1() const {
        return offset1;
    }

    /**
     * @return the x/y size (extent) in pixels of the first region.
     */
    const Region& getRegion1() const {
        return region1;
    }

    /**
     * @return the channel that is used for the first region. only used for color featurse.
     */
    uint8_t getChannel1() const {
        return channel1;
    }

    /**
     * @return the x/y offset in pixels of the second region.
     */
    const Offset& getOffset2() const {
        return offset2;
    }

    /**
     * @return the x/y size (extent) in pixels of the second region.
     */
    const Region& getRegion2() const {
        return region2;
    }

    /**
     * @return the channel that is used for the second region. only used for color featurse.
     */
    uint8_t getChannel2() const {
        return channel2;
    }

    /**
     * @return whether this image feature is not equal to another
     */
    bool operator!=(const ImageFeatureFunction& other) const {
        return !(*this == other);
    }
 
    /**
     * @return whether this image feature is equal to another
     */
    bool operator==(const ImageFeatureFunction& other) const;

private:
    FeatureType featureType;

    Offset offset1;
    Region region1;
    uint8_t channel1;

    Offset offset2;
    Region region2;
    uint8_t channel2;

    FeatureResponseType calculateColorFeature(const PixelInstance& instance, bool flipRegion) const {

        const Depth depth = instance.getDepth();
        if (!depth.isValid()) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        FeatureResponseType a;
        if (flipRegion)
        	a = instance.averageRegionColor(Offset(-offset1.getX(),offset1.getY()).normalize(depth), region1.normalize(depth),
                    channel1);
        else
        	a = instance.averageRegionColor(offset1.normalize(depth), region1.normalize(depth),
                channel1);
        if (isnan(a))
            return a;

        FeatureResponseType b;
        if (flipRegion)
        	b = instance.averageRegionColor(Offset(-offset2.getX(),offset2.getY()).normalize(depth), region2.normalize(depth),
                    channel2);
        else
        	b = instance.averageRegionColor(offset2.normalize(depth), region2.normalize(depth),
                channel2);
        if (isnan(b))
            return b;

        return (a - b);
    }

    FeatureResponseType calculateDepthFeature(const PixelInstance& instance, bool flipRegion) const {

        const Depth depth = instance.getDepth();
        if (!depth.isValid()) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        FeatureResponseType a;
        if (flipRegion)
        	a = instance.averageRegionDepth(Offset(-offset1.getX(),offset1.getY()).normalize(depth), region1.normalize(depth));
        else
        	a = instance.averageRegionDepth(offset1.normalize(depth), region1.normalize(depth));
        if (isnan(a)) {
            return a;
        }

        FeatureResponseType b;
        if (flipRegion)
        	b = instance.averageRegionDepth(Offset(-offset2.getX(),offset2.getY()).normalize(depth), region2.normalize(depth));
        else
        	b = instance.averageRegionDepth(offset2.normalize(depth), region2.normalize(depth));
        if (isnan(b)) {
            return b;
        }

        assert(a > 0);
        assert(b > 0);

        return (a - b);
    }

};

/**
 * Helper class to store a list of features and a list of threshold per feature in a compact manner
 * @ingroup transfer_cpu_gpu
 *
 * such that it can be transferred between CPU and GPU.
 *
 * Clients are not intended to use this class directly.
 */
template<class memory_space>
class ImageFeaturesAndThresholds {

public:

    cuv::ndarray<int8_t, memory_space> m_features; /**< array of features */
    cuv::ndarray<float, memory_space> m_thresholds; /**< array of thresholds */

private:

    explicit ImageFeaturesAndThresholds(const cuv::ndarray<int8_t, memory_space>& features,
            const cuv::ndarray<float, memory_space>& thresholds) :
            m_features(features), m_thresholds(thresholds) {
    }

public:

    /**
     * set the size of the features and thresholds arrays
     */
    explicit ImageFeaturesAndThresholds(size_t numFeatures, size_t numThresholds,
            boost::shared_ptr<cuv::allocator> allocator) :
            m_features(11, numFeatures, allocator), m_thresholds(numThresholds, numFeatures, allocator) {
    }

    /**
     * copy the features and thresholds of another object
     */
    template<class other_memory_space>
    explicit ImageFeaturesAndThresholds(const ImageFeaturesAndThresholds<other_memory_space>& other) :
            m_features(other.features().copy()), m_thresholds(other.thresholds().copy()) {
    }

    /**
     * set the features and thresholds using another object attributes
     */
    template<class other_memory_space>
    ImageFeaturesAndThresholds& operator=(const ImageFeaturesAndThresholds<other_memory_space>& other) {
        m_features = other.features().copy();
        m_thresholds = other.thresholds().copy();
        return (*this);
    }

    /**
     * @return a copy of the object
     */
    ImageFeaturesAndThresholds copy() const {
        return ImageFeaturesAndThresholds(m_features.copy(), m_thresholds.copy());
    }

    /**
     * @return features - const
     */
    const cuv::ndarray<int8_t, memory_space> features() const {
        return m_features;
    }

    /**
     * @return features
     */
    cuv::ndarray<int8_t, memory_space> features() {
        return m_features;
    }

    /**
     * @return feature types
     */    
    cuv::ndarray<int8_t, memory_space> types() {
        return m_features[cuv::indices[0][cuv::index_range()]];
    }

    /**
     * @return x offsets of the first region 
     */
    cuv::ndarray<int8_t, memory_space> offset1X() {
        return m_features[cuv::indices[1][cuv::index_range()]];
    }

    /**
     * @return y offsets of the first region
     */
    cuv::ndarray<int8_t, memory_space> offset1Y() {
        return m_features[cuv::indices[2][cuv::index_range()]];
    }

    /**
     * @return x offsets of the second region
     */
    cuv::ndarray<int8_t, memory_space> offset2X() {
        return m_features[cuv::indices[3][cuv::index_range()]];
    }

    /**
     * @return y offsets of the second region
     */
    cuv::ndarray<int8_t, memory_space> offset2Y() {
        return m_features[cuv::indices[4][cuv::index_range()]];
    }

    /**
     * @return widths of the first region
     */
    cuv::ndarray<int8_t, memory_space> region1X() {
        return m_features[cuv::indices[5][cuv::index_range()]];
    }

    /**
     * @return heights of the first region
     */
    cuv::ndarray<int8_t, memory_space> region1Y() {
        return m_features[cuv::indices[6][cuv::index_range()]];
    }

    /**
     * @return widths of the second region
     */
    cuv::ndarray<int8_t, memory_space> region2X() {
        return m_features[cuv::indices[7][cuv::index_range()]];
    }

    /**
     * @return heights of the second region
     */
    cuv::ndarray<int8_t, memory_space> region2Y() {
        return m_features[cuv::indices[8][cuv::index_range()]];
    }

    /**
     * @return features first channels
     */
    cuv::ndarray<int8_t, memory_space> channel1() {
        return m_features[cuv::indices[9][cuv::index_range()]];
    }

    /**
     * @return features second channels
     */
    cuv::ndarray<int8_t, memory_space> channel2() {
        return m_features[cuv::indices[10][cuv::index_range()]];
    }

    /**
     * @return thresholds
     */
    cuv::ndarray<float, memory_space> thresholds() {
        return this->m_thresholds;
    }

    /**
     * @return feature types - const
     */
    const cuv::ndarray<int8_t, memory_space> types() const {
        return m_features[cuv::indices[0][cuv::index_range()]];
    }

    /**
     * @return x offsets of the first region - const
     */
    const cuv::ndarray<int8_t, memory_space> offset1X() const {
        return m_features[cuv::indices[1][cuv::index_range()]];
    }

    /**
     * @return y offsets of the first region - const
     */
    const cuv::ndarray<int8_t, memory_space> offset1Y() const {
        return m_features[cuv::indices[2][cuv::index_range()]];
    }

    /**
     * @return x offsets of the second region - const
     */
    const cuv::ndarray<int8_t, memory_space> offset2X() const {
        return m_features[cuv::indices[3][cuv::index_range()]];
    }

    /**
     * @return y offsets of the second region - const
     */
    const cuv::ndarray<int8_t, memory_space> offset2Y() const {
        return m_features[cuv::indices[4][cuv::index_range()]];
    }

    /**
     * @return widths of the first region - const
     */
    const cuv::ndarray<int8_t, memory_space> region1X() const {
        return m_features[cuv::indices[5][cuv::index_range()]];
    }

    /**
     * @return heights of the first region - const
     */
    const cuv::ndarray<int8_t, memory_space> region1Y() const {
        return m_features[cuv::indices[6][cuv::index_range()]];
    }

    /**
     * @return widths of the second region - const
     */
    const cuv::ndarray<int8_t, memory_space> region2X() const {
        return m_features[cuv::indices[7][cuv::index_range()]];
    }

    /**
     * @return heights of the second region - const
     */
    const cuv::ndarray<int8_t, memory_space> region2Y() const {
        return m_features[cuv::indices[8][cuv::index_range()]];
    }

    /**
     * @return features first channels - const
     */
    const cuv::ndarray<int8_t, memory_space> channel1() const {
        return m_features[cuv::indices[9][cuv::index_range()]];
    }

    /**
     * @return features second channels - const
     */
    const cuv::ndarray<int8_t, memory_space> channel2() const {
        return m_features[cuv::indices[10][cuv::index_range()]];
    }

    /**
     * @return thresholds - const
     */
    const cuv::ndarray<float, memory_space> thresholds() const {
        return this->m_thresholds;
    }

    /**
     * @return threshold of the given feature
     */
    double getThreshold(size_t threshNr, size_t featNr) const {
        return m_thresholds(threshNr, featNr);
    }

    /**
     * fill the feature attributes using the given feature at the specified location
     */
    void setFeatureFunction(size_t feat, const ImageFeatureFunction& feature) {

        types()(feat) = static_cast<int8_t>(feature.getType());

        offset1X()(feat) = feature.getOffset1().getX();
        offset1Y()(feat) = feature.getOffset1().getY();
        offset2X()(feat) = feature.getOffset2().getX();
        offset2Y()(feat) = feature.getOffset2().getY();

        region1X()(feat) = feature.getRegion1().getX();
        region1Y()(feat) = feature.getRegion1().getY();
        region2X()(feat) = feature.getRegion2().getX();
        region2Y()(feat) = feature.getRegion2().getY();

        channel1()(feat) = feature.getChannel1();
        channel2()(feat) = feature.getChannel2();

        assert(getFeatureFunction(feat) == feature);
    }

    /**
     * @return feature that has feat as number
     */
    ImageFeatureFunction getFeatureFunction(size_t feat) const {
        const Offset offset1(offset1X()(feat), offset1Y()(feat));
        const Offset offset2(offset2X()(feat), offset2Y()(feat));
        const Offset region1(region1X()(feat), region1Y()(feat));
        const Offset region2(region2X()(feat), region2Y()(feat));
        return ImageFeatureFunction(static_cast<FeatureType>(static_cast<int8_t>(types()(feat))),
                offset1, region1,
                channel1()(feat),
                offset2, region2,
                channel2()(feat));
    }

};

/**
 * Helper class to store a list of pixel instances in a compact manner
 * @ingroup transfer_cpu_gpu
 *
 * such that it can be transferred between CPU and GPU.
 *
 * Clients are not intended to use this class directly.
 */
template<class memory_space>
class Samples {

public:
    cuv::ndarray<int, memory_space> data;  /**< all data associated with the pixel */

    float* depths; /**< depth of the pixel */
    int* sampleX; /**< x coordinate of the pixel */
    int* sampleY; /**< y coordinate of the pixel */
    int* imageNumbers; /**< number of image that has the pixel */
    uint8_t* labels; /**< label of the pixel */
    HorizontalFlipSetting* horFlipSetting; /**< flipping setting of the pixel */

    /**
     * does not copy data
     */
    Samples(const Samples& samples) :
            data(samples.data),
                    depths(reinterpret_cast<float*>(data[cuv::indices[0][cuv::index_range()]].ptr())),
                    sampleX(reinterpret_cast<int*>(data[cuv::indices[1][cuv::index_range()]].ptr())),
                    sampleY(reinterpret_cast<int*>(data[cuv::indices[2][cuv::index_range()]].ptr())),
                    imageNumbers(reinterpret_cast<int*>(data[cuv::indices[3][cuv::index_range()]].ptr())),
                    labels(reinterpret_cast<uint8_t*>(data[cuv::indices[4][cuv::index_range()]].ptr())),
                    horFlipSetting(reinterpret_cast<HorizontalFlipSetting*>(data[cuv::indices[5][cuv::index_range()]].ptr())){
    }

    /**
     *  copies the data
     */
    template<class T>
    Samples(const Samples<T>& samples, cudaStream_t stream) :
            data(samples.data, stream),
                    depths(reinterpret_cast<float*>(data[cuv::indices[0][cuv::index_range()]].ptr())),
                    sampleX(reinterpret_cast<int*>(data[cuv::indices[1][cuv::index_range()]].ptr())),
                    sampleY(reinterpret_cast<int*>(data[cuv::indices[2][cuv::index_range()]].ptr())),
                    imageNumbers(reinterpret_cast<int*>(data[cuv::indices[3][cuv::index_range()]].ptr())),
                    labels(reinterpret_cast<uint8_t*>(data[cuv::indices[4][cuv::index_range()]].ptr())),
                    horFlipSetting(reinterpret_cast<HorizontalFlipSetting*>(data[cuv::indices[5][cuv::index_range()]].ptr())){
    }

    /**
     * allocates memory for data
     */
    Samples(size_t numSamples, boost::shared_ptr<cuv::allocator>& allocator) :
            data(6, numSamples, allocator),
                    depths(reinterpret_cast<float*>(data[cuv::indices[0][cuv::index_range()]].ptr())),
                    sampleX(reinterpret_cast<int*>(data[cuv::indices[1][cuv::index_range()]].ptr())),
                    sampleY(reinterpret_cast<int*>(data[cuv::indices[2][cuv::index_range()]].ptr())),
                    imageNumbers(reinterpret_cast<int*>(data[cuv::indices[3][cuv::index_range()]].ptr())),
                    labels(reinterpret_cast<uint8_t*>(data[cuv::indices[4][cuv::index_range()]].ptr())),
                    horFlipSetting(reinterpret_cast<HorizontalFlipSetting*>(data[cuv::indices[5][cuv::index_range()]].ptr()))
    {
        assert_equals(imageNumbers, data.ptr() + 3 * numSamples);
        assert_equals(labels, reinterpret_cast<uint8_t*>(data.ptr() + 4 * numSamples));
    }

};

/**
 * Helper class for the four phases in the cost-intensive best-split evaluation during random forest training.
 * See the Master’s thesis "Accelerating Random Forests on CPUs and GPUs for Object-Class Image Segmentation"
 * for more details on this implementation.
 * @ingroup training_helper_classes
 *
 * Clients are not intended to use this class directly.
 */
class ImageFeatureEvaluation {
public:
    // box_radius: > 0, half the box side length to uniformly sample
    //    (dx,dy) offsets from.

	/**
	 * helper object for feature evaluation for the given tree and training configuration
	 */
    ImageFeatureEvaluation(const size_t treeId, const TrainingConfiguration& configuration) :
            treeId(treeId), configuration(configuration),
                    imageWidth(0), imageHeight(0),
                    sampleDataAllocator(boost::make_shared<cuv::pooled_cuda_allocator>("sampleData")),
                    featuresAllocator(boost::make_shared<cuv::pooled_cuda_allocator>("feature")),
                    keysIndicesAllocator(boost::make_shared<cuv::pooled_cuda_allocator>("keysIndices")),
                    scoresAllocator(boost::make_shared<cuv::pooled_cuda_allocator>("scores")),
                    countersAllocator(boost::make_shared<cuv::pooled_cuda_allocator>("counters")),
                    featureResponsesAllocator(boost::make_shared<cuv::pooled_cuda_allocator>("featureResponses")) {
        assert(configuration.getBoxRadius() > 0);
        assert(configuration.getRegionSize() > 0);

        initDevice();
    }

    /**
     * @return best features and thresholds to split after evaluating splits
     */
    std::vector<SplitFunction<PixelInstance, ImageFeatureFunction> > evaluateBestSplits(RandomSource& randomSource,
            const std::vector<std::pair<boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> >,
                    std::vector<const PixelInstance*> > >& samplesPerNode);

    /**
     * @return a batch that contains the given samples
     */
    std::vector<std::vector<const PixelInstance*> > prepare(const std::vector<const PixelInstance*>& samples,
            RandomTree<PixelInstance, ImageFeatureFunction>& node, cuv::host_memory_space);

    /**
     * @return a batch that contains the given samples
     */
    std::vector<std::vector<const PixelInstance*> > prepare(const std::vector<const PixelInstance*>& samples,
            RandomTree<PixelInstance, ImageFeatureFunction>& node, cuv::dev_memory_space, bool keepMutexLocked = true);

    /**
     * @return random features and thresholds
     */
    ImageFeaturesAndThresholds<cuv::host_memory_space> generateRandomFeatures(
            const std::vector<const PixelInstance*>& batches,
            int seed, const bool sort, cuv::host_memory_space);

    /**
     * @return random features and thresholds
     */
    ImageFeaturesAndThresholds<cuv::dev_memory_space> generateRandomFeatures(
            const std::vector<const PixelInstance*>& batches,
            int seed, const bool sort, cuv::dev_memory_space);

    /**
     * sort features by the given keys
     */
    template<class memory_space>
    void sortFeatures(ImageFeaturesAndThresholds<memory_space>& featuresAndThresholds,
            const cuv::ndarray<int, memory_space>& keysIndices) const;

    /**
     * @return histogram counters after calculating feature responses
     */
    template<class memory_space>
    cuv::ndarray<WeightType, memory_space> calculateFeatureResponsesAndHistograms(
            RandomTree<PixelInstance, ImageFeatureFunction>& node,
            const std::vector<std::vector<const PixelInstance*> >& batches,
            const ImageFeaturesAndThresholds<memory_space>& featuresAndThresholds,
            cuv::ndarray<FeatureResponseType, cuv::host_memory_space>* featureResponsesHost = 0);

    /**
     * @return normalized cores after calculating the information gain that results from splitting
     */
    template<class memory_space>
    cuv::ndarray<ScoreType, cuv::host_memory_space> calculateScores(
            const cuv::ndarray<WeightType, memory_space>& counters,
            const ImageFeaturesAndThresholds<memory_space>& featuresAndThresholds,
            const cuv::ndarray<WeightType, memory_space>& histogram);

private:

    void selectDevice();

    void initDevice();

    void copyFeaturesToDevice();

    Samples<cuv::dev_memory_space> copySamplesToDevice(const std::vector<const PixelInstance*>& samples,
            cudaStream_t stream);

    const ImageFeatureFunction sampleFeature(RandomSource& randomSource,
            const std::vector<const PixelInstance*>&) const;

    const size_t treeId;
    const TrainingConfiguration& configuration;

    unsigned int imageWidth;
    unsigned int imageHeight;

    boost::shared_ptr<cuv::allocator> sampleDataAllocator;
    boost::shared_ptr<cuv::allocator> featuresAllocator;
    boost::shared_ptr<cuv::allocator> keysIndicesAllocator;
    boost::shared_ptr<cuv::allocator> scoresAllocator;
    boost::shared_ptr<cuv::allocator> countersAllocator;
    boost::shared_ptr<cuv::allocator> featureResponsesAllocator;
};

/**
 * A random tree in a random forest, for RGB-D images.
 * @ingroup forest_hierarchy
 */
class RandomTreeImage {
public:

    /**
     * @param id tree number
     * @param configuration training configuration of the tree
     */
    RandomTreeImage(int id, const TrainingConfiguration& configuration);

    /**
     * @param tree
     * @param configuration training configuration of the tree
     * @param classLabelPriorDistribution prior distributions of the classes
     */
    RandomTreeImage(boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> > tree,
            const TrainingConfiguration& configuration,
            const cuv::ndarray<WeightType, cuv::host_memory_space>& classLabelPriorDistribution);

    /**
     * Trains the tree from the given training images
     */
    void train(const std::vector<LabeledRGBDImage>& trainLabelImages,
            RandomSource& randomSource, size_t subsampleCount, size_t numLabels);

    /**
     * Predicts the given image and outputs the prediction.
     */
    void test(const RGBDImage* image, LabelImage& prediction) const;

    /**
     * Normalize the leaf-node histograms in the tree
     *
     * @param histogramBias the histogram bias for the normalization
     */
    void normalizeHistograms(const double histogramBias);

    /**
     * @return the underlying tree
     */
    const boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> >& getTree() const {
        return tree;
    }

    /**
     * @return prior distribution of classes
     */
    const cuv::ndarray<WeightType, cuv::host_memory_space>& getClassLabelPriorDistribution() const {
        return classLabelPriorDistribution;
    }

    /**
     * @return tree id
     */
    size_t getId() const {
        return id;
    }

    /**
     * @return whether the given label is one that the user specified as being ignored
     */
    bool shouldIgnoreLabel(const LabelType& label) const;

private:

    void doTrain(RandomSource& randomSource, size_t numClasses,
            std::vector<const PixelInstance*>& subsamples);

    bool finishedTraining;
    size_t id;

    const TrainingConfiguration configuration;

    boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> > tree;

    cuv::ndarray<WeightType, cuv::host_memory_space> classLabelPriorDistribution;

    void calculateLabelPriorDistribution(const std::vector<LabeledRGBDImage>& trainLabelImages);

    std::vector<PixelInstance> subsampleTrainingDataPixelUniform(
            const std::vector<LabeledRGBDImage>& trainLabelImages,
            RandomSource& randomSource, size_t subsampleCount) const;

    std::vector<PixelInstance> subsampleTrainingDataClassUniform(
            const std::vector<LabeledRGBDImage>& trainLabelImages,
            RandomSource& randomSource, size_t subsampleCount) const;

};

}

std::ostream& operator<<(std::ostream& os, const curfil::RandomTreeImage& tree);

std::ostream& operator<<(std::ostream& os, const curfil::ImageFeatureFunction& featureFunction);

std::ostream& operator<<(std::ostream& os, const curfil::XY& xy);

#endif

