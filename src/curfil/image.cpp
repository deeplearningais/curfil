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
#include "image.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <cmath>
#include <iomanip>
#include <map>
#include <string>
#include <tbb/mutex.h>
#include <tbb/parallel_for.h>
#include <vigra/colorconversions.hxx>
#include <vigra/imageinfo.hxx>
#include <vigra/impex.hxx>
#include <vigra/transformimage.hxx>

#include "utils.h"

namespace fs = boost::filesystem;

namespace curfil {

const Depth Depth::INVALID = Depth(-1);

RGBColor::RGBColor() :
        std::vector<uint8_t>(3, 0) {
}

RGBColor::RGBColor(uint8_t r, uint8_t g, uint8_t b) :
        std::vector<uint8_t>(3, 0) {
    (*this)[0] = r;
    (*this)[1] = g;
    (*this)[2] = b;
}

RGBColor::RGBColor(const std::string& colorString) :
                std::vector<uint8_t>(3, 0) {
            std::vector<std::string> strs;
            // key is in the format: "255,255,255"
            boost::split(strs, colorString, boost::is_any_of(","));

            try {
                int r = boost::lexical_cast<int>(strs[0]);
                int g = boost::lexical_cast<int>(strs[1]);
                int b = boost::lexical_cast<int>(strs[2]);

        if (r < std::numeric_limits<uint8_t>::min() || r > std::numeric_limits<uint8_t>::max()
                || g < std::numeric_limits<uint8_t>::min() || g > std::numeric_limits<uint8_t>::max()
                || b < std::numeric_limits<uint8_t>::min() || b > std::numeric_limits<uint8_t>::max()) {
            throw std::runtime_error(std::string("color value out of range [0-255]: ") + colorString);
        }

        std::vector<uint8_t> color(3, 0);
        (*this)[0] = static_cast<uint8_t>(r);
        (*this)[1] = static_cast<uint8_t>(g);
        (*this)[2] = static_cast<uint8_t>(b);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("illegal color: '") + colorString + "': " + e.what());
    }
}

static vigra::DVector3Image convertRGB2CIELab(const vigra::DVector3Image& srcImage) {
    vigra::DVector3Image dstImage(srcImage.width(), srcImage.height());

    vigra::transformImage(srcImageRange(srcImage), destImage(dstImage),
            vigra::RGB2LabFunctor<double>());

    return dstImage;
}

static vigra::DVector3Image convertCIELab2RGB(const vigra::DVector3Image& srcImage) {
    vigra::DVector3Image dstImage(srcImage.width(), srcImage.height());

    vigra::transformImage(srcImageRange(srcImage), destImage(dstImage),
            vigra::Lab2RGBFunctor<double>());

    return dstImage;
}

template<class T>
static void loadImage(const std::string& filename, T& image) {
    vigra::ImageImportInfo info(filename.c_str());

    if (info.isGrayscale() || !info.isColor() || info.numBands() != 3) {
        throw std::runtime_error("loading of non-RGB images is not yet supported");
    }

    vigra::Size2D size = info.size();
    image = T(size.width(), size.height());

    vigra::importImage(info, vigra::destImage(image));
}

RGBDImage::RGBDImage(const std::string& filename, const std::string& depthFilename, bool useDepthImages, bool convertToCIELab,
        bool useDepthFilling, bool calculateIntegralImage) :
        filename(filename), depthFilename(depthFilename),
                colorImage(boost::make_shared<cuv::cuda_allocator>()),
                depthImage(boost::make_shared<cuv::cuda_allocator>()),
                inCIELab(false), integratedColor(false), integratedDepth(false) {

    {
        utils::Profile profile("loadImage");

        vigra::DVector3Image image;
        try {
            loadImage(filename, image);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("failed to load image '") + filename + "': " + e.what());
        }

        if (convertToCIELab) {
            image = convertRGB2CIELab(image);
        }

        width = image.width();
        height = image.height();
        assert(width >= 0 && height >= 0);
        colorImage.resize(cuv::extents[COLOR_CHANNELS][getHeight()][getWidth()]);
        for (int y = 0; y < getHeight(); ++y) {
            for (int x = 0; x < getWidth(); ++x) {
                for (unsigned int c = 0; c < COLOR_CHANNELS; ++c) {
                    setColor(x, y, c, image(x, y)[c]);
                }
            }
        }

        inCIELab = convertToCIELab;

		if (useDepthImages) {
			try {
				loadDepthImage(depthFilename);
			} catch (const std::exception& e) {
				throw std::runtime_error(
						std::string("failed to load depth image '")
								+ depthFilename + "': " + e.what());
			}
		} else
			loadDummyDepthValues();

    }

    if (useDepthFilling) {
        fillDepth();
    }

    if (calculateIntegralImage) {
        calculateIntegral();
    }
}

RGBDImage::RGBDImage(const RGBDImage& other) :
        filename(other.filename), depthFilename(other.depthFilename),
                width(other.width), height(other.height),
                colorImage(other.colorImage.copy()),
                depthImage(other.depthImage.copy()),
                inCIELab(other.inCIELab), integratedColor(other.integratedColor), integratedDepth(other.integratedDepth) {
}

template<class A, class B>
static void copy(A* dst, const B* src, size_t size) {
    for (size_t i = 0; i < size; i++) {
        dst[i] = src[i];
    }
}

template<class A, class B>
static void copy(cuv::ndarray<A, cuv::host_memory_space>& dst, const cuv::ndarray<B, cuv::host_memory_space>& src) {
    assert(dst.size() == src.size());
    assert(dst.shape() == src.shape());
    copy(dst.ptr(), src.ptr(), dst.size());
}

// derives image in double precision
void RGBDImage::calculateDerivative(cuv::ndarray_view<float, cuv::host_memory_space>& data) {

    assert(data.ndim() == 2);

    cuv::ndarray<double, cuv::host_memory_space> tempData(data.shape());
    copy(tempData, data);

    cuv::ndarray_view<double, cuv::host_memory_space> view =
            tempData[cuv::indices[cuv::index_range()][cuv::index_range()]];

    calculateDerivative<double>(view);

    copy(data, tempData);
}

template<class T>
void RGBDImage::calculateDerivative(cuv::ndarray_view<T, cuv::host_memory_space>& view) {
    assert(view.ndim() == 2);
    const int height = view.shape(0);
    const int width = view.shape(1);
    T* data = view.ptr();
    for (int y = height - 1; y >= 0; --y) {
        const size_t previousRow = (y - 1) * width;
        const size_t rowOffset = y * width;
        for (int x = width - 1; x >= 0; --x) {
            T above = (y > 0) ? static_cast<T>(data[previousRow + x]) : T();
            T left = (x > 0) ? static_cast<T>(data[rowOffset + x - 1]) : T();
            T aboveLeft = (x > 0 && y > 0) ? static_cast<T>(data[previousRow + x - 1]) : T();

            data[rowOffset + x] -= left + above - aboveLeft;
        }
    }
}

// integrates channel in double precision
void RGBDImage::calculateIntegral(cuv::ndarray_view<float, cuv::host_memory_space>& data) {

    assert(data.ndim() == 2);

    cuv::ndarray<double, cuv::host_memory_space> tempData(data.shape());
    copy(tempData, data);

    cuv::ndarray_view<double, cuv::host_memory_space> view =
            tempData[cuv::indices[cuv::index_range()][cuv::index_range()]];
    calculateIntegral<double>(view);

    copy(data, tempData);
}

template<class T>
void RGBDImage::calculateIntegral(cuv::ndarray_view<T, cuv::host_memory_space>& view) {

    assert(view.ndim() == 2);
    const int height = view.shape(0);
    const int width = view.shape(1);
    T* data = view.ptr();

    // Kahan summation algorithm
    // http://en.wikipedia.org/wiki/Kahan_summation_algorithm
    double c = 0.0;

    for (int y = 0; y < height; ++y) {
        const int previousRow = (y - 1) * width;
        const int rowOffset = y * width;
        for (int x = 0; x < width; ++x) {
            T above = (y > 0) ? static_cast<T>(data[previousRow + x]) : T();
            T left = (x > 0) ? static_cast<T>(data[rowOffset + x - 1]) : T();
            T aboveLeft = (x > 0 && y > 0) ? static_cast<T>(data[previousRow + x - 1]) : T();

            double dy = (left + above - aboveLeft) - c;
            double t = data[rowOffset + x] + dy;
            c = (t - data[rowOffset + x]) - dy;

            data[rowOffset + x] = t;
        }
    }
}

void RGBDImage::fillDepth() {
    if (integratedDepth) {
        throw std::runtime_error("can not fill depth on integrated depth");
    }

    utils::Profile profile("depthFilling");

    fillDepthFromRight();
    fillDepthFromLeft();
    fillDepthFromTop();
    fillDepthFromBottom();

    // update depth valids
    int* depths = depthImage.ptr();
    int* depthValids = depthImage.ptr() + 1 * getWidth() * getHeight();
    for (int y = 0; y < getHeight(); y++) {
        const size_t rowOffset = y * getWidth();
        for (int x = 0; x < getWidth(); x++) {
            depthValids[rowOffset + x] = static_cast<int>(depths[rowOffset + x] > 0);
        }
    }

#ifndef NDEBUG
    for (int y = 0; y < getHeight(); y++) {
        for (int x = 0; x < getWidth(); x++) {
            assert(getDepth(x, y).isValid());
            assert(getDepthValid(x, y) == 1);
        }
    }
#endif

}

void RGBDImage::fillDepthFromRight() {
    if (integratedDepth) {
        throw std::runtime_error("can not fill depth on integrated depth");
    }

    int* depths = depthImage.ptr();
    for (int y = 0; y < getHeight(); y++) {
        const size_t rowOffset = y * getWidth();
        for (int x = getWidth() - 2; x >= 0; x--) {
            int& depth = depths[rowOffset + x];
            if (!depth) {
                depth = depths[rowOffset + x + 1];
            }
        }
    }
}

void RGBDImage::fillDepthFromLeft() {
    if (integratedDepth) {
        throw std::runtime_error("can not fill depth on integrated depth");
    }
    int* depths = depthImage.ptr();
    for (int y = 0; y < getHeight(); y++) {
        const size_t rowOffset = y * getWidth();
        for (int x = 1; x < getWidth(); x++) {
            int& depth = depths[rowOffset + x];
            if (!depth) {
                depth = depths[rowOffset + x - 1];
            }
        }
    }
}

void RGBDImage::fillDepthFromTop() {
    if (integratedDepth) {
        throw std::runtime_error("can not fill depth on integrated depth");
    }
    int* depths = depthImage.ptr();
    for (int y = 1; y < getHeight(); y++) {
        const size_t rowOffset = y * getWidth();
        const size_t previousRow = (y - 1) * getWidth();
        for (int x = 0; x < getWidth(); x++) {
            int& depth = depths[rowOffset + x];
            if (!depth) {
                depth = depths[previousRow + x];
            }
        }
    }
}

void RGBDImage::fillDepthFromBottom() {
    if (integratedDepth) {
        throw std::runtime_error("can not fill depth on integrated depth");
    }
    int* depths = depthImage.ptr();
    for (int y = getHeight() - 2; y >= 0; y--) {
        const size_t rowOffset = y * getWidth();
        const size_t nextRow = (y + 1) * getWidth();
        for (int x = 0; x < getWidth(); x++) {
            int& depth = depths[rowOffset + x];
            if (!depth) {
                depth = depths[nextRow + x];
            }
        }
    }
}

void RGBDImage::loadDummyDepthValues() {

	depthImage.resize(cuv::extents[DEPTH_CHANNELS][getHeight()][getWidth()]);

	int* depths = depthImage.ptr();
	int* depthValid = depthImage.ptr() + getWidth() * getHeight();

	for (int y = 0; y < getHeight(); y++) {
		const size_t rowOffset = y * getWidth();
		for (int x = 0; x < getWidth(); x++) {
			depths[rowOffset + x] = 1000;
			depthValid[rowOffset + x] = 1;
		}
	}

}

void RGBDImage::loadDepthImage(const std::string& depthFilename) {

    vigra::ImageImportInfo info(depthFilename.c_str());

    if (!info.isGrayscale() || info.isColor() || info.numBands() != 1) {
        throw std::runtime_error("invalid depth image format");
    }

    vigra::Size2D size = info.size();

    if (size.width() != getWidth()) {
        throw std::runtime_error(
                (boost::format("width of color and depth image differ: %d vs. %d") % getWidth() % size.width()).str());
    }
    if (size.height() != getHeight()) {
        throw std::runtime_error(
                (boost::format("height of color and depth image differ: %d vs. %d") % getHeight() % size.height()).str());
    }

    vigra::UInt16Image image(size.width(), size.height());
    vigra::UInt16Image image2(size.width(), size.height()); // zero-initialized by default.

    size_t idx = depthFilename.find("_depth");
    if(idx != std::string::npos){
      std::string heightFilename = depthFilename;
      heightFilename.replace(idx, 6, "_height");
      if(fs::exists(heightFilename)){
        vigra::ImageImportInfo info2(heightFilename.c_str());
        vigra::importImage(info2, vigra::destImage(image2));
        std::cout << "using height from " << heightFilename <<std::endl;
      }
    }
    else
      std::cout << "not using height for " << depthFilename <<std::endl;
    vigra::importImage(info, vigra::destImage(image));

    depthImage.resize(cuv::extents[DEPTH_CHANNELS][getHeight()][getWidth()]);

    utils::Average depthAverage;

    int* depths     = depthImage.ptr();
    int* depthValid = depthImage.ptr() + getWidth() * getHeight();
    int* heights    = depthValid + getWidth() * getHeight();

    for (int y = 0; y < getHeight(); y++) {
        const size_t rowOffset = y * getWidth();
        for (int x = 0; x < getWidth(); x++) {
            const int depth = image(x, y);
            const int height = image(x, y);
            if (depth < 0 || depth > 50000) {
                throw std::runtime_error((boost::format("illegal depth value in image %s @%d,%d: %d")
                        % depthFilename % x % y % depth).str());
            }
            if (height < 0 || height > 50000) {
              throw std::runtime_error((boost::format("illegal height value in image %s @%d,%d: %d")
                                        % depthFilename % x % y % depth).str());
            }

            if (depth > 0) {
                depthAverage.addValue(depth / 1000.0);
            }

            depths[rowOffset + x] = depth;
            heights[rowOffset + x] = height;
            depthValid[rowOffset + x] = static_cast<int>(depth > 0);
        }
    }

    double depthAvg = depthAverage.getAverage();
    if (depthAvg < 0.01 || depthAvg > 10.0) {
        throw std::runtime_error((boost::format("illegal average depth of '%s': %e") % depthFilename % depthAvg).str());
    }
}

void RGBDImage::resizeImage(int newWidth, int newHeight)
{
	int originalWidth = getWidth();
	int originalHeight = getHeight();
	using namespace cuv;
	ndarray<float, host_memory_space>  ci(extents[COLOR_CHANNELS][newHeight][newWidth]);
	ndarray<int, host_memory_space>  di(extents[DEPTH_CHANNELS][newHeight][newWidth]);

	ci = std::numeric_limits<float>::quiet_NaN();

	for (unsigned int c = 0; c < COLOR_CHANNELS; ++c) {
		ci[indices[c][index_range(0, originalHeight)][index_range(0, originalWidth)]]  = colorImage[indices[c][index_range()][index_range()]];
	}
	colorImage = ci;

	//di = std::numeric_limits<int>::quiet_NaN();
	di = 0;

	for (unsigned int d = 0; d < DEPTH_CHANNELS; ++d) {
		di[indices[d][index_range(0, originalHeight)][index_range(0, originalWidth)]] = depthImage[indices[d][index_range()][index_range()]];
	}
	depthImage = di;

	width = newWidth;
	height = newHeight;
}

void RGBDImage::dump(std::ostream& out) const {

    double max[3] = { 0, 0, 0 };
    for (unsigned int c = 0; c < COLOR_CHANNELS; ++c) {
        for (int y = 0; y < getHeight(); ++y) {
            for (int x = 0; x < getWidth(); ++x) {
                max[c] = std::max(max[c],
                        std::ceil(std::log(getColor(x, y, c)) / std::log(10)));
            }
        }
    }

    for (unsigned int c = 0; c < COLOR_CHANNELS; ++c) {
        for (int y = 0; y < getHeight(); ++y) {
            for (int x = 0; x < getWidth(); ++x) {
                if (c > 0) {
                    out << ",";
                }
                out << std::setw(max[c]) << getColor(x, y, c);
            }
            out << "  ";
        }
        out << std::endl;
    }
}

void RGBDImage::dumpDepth(std::ostream& out) const {
    for (int y = 0; y < getHeight(); ++y) {
        for (int x = 0; x < getWidth(); ++x) {
            out.width(5);
            const Depth& depth = getDepth(x, y);
            out << (depth.isValid() ? depth.getFloatValue() : 0);
        }
        out << std::endl;
    }
}

void RGBDImage::dumpDepthValid(std::ostream& out) const {
    for (int y = 0; y < getHeight(); ++y) {
        for (int x = 0; x < getWidth(); ++x) {
            out << getDepthValid(x, y);
            out << "   ";
        }
        out << std::endl;
    }
}

void RGBDImage::reset() {
    depthImage = 0;
    colorImage = 0.0f;
    integratedColor = false;
    integratedDepth = false;
}

void RGBDImage::calculateIntegral() {

    utils::Profile profile("calculateImageIntegral");

    if (integratedColor || integratedDepth) {
        throw std::runtime_error("image already integrated");
    }

    tbb::parallel_for(tbb::blocked_range<size_t>(0, COLOR_CHANNELS + DEPTH_CHANNELS, 1),
            [&](const tbb::blocked_range<size_t>& range) {
                for(unsigned int channelNr = range.begin(); channelNr != range.end(); channelNr++) {
                    if (channelNr >= COLOR_CHANNELS) {
                        unsigned int depthChannelNr = channelNr - COLOR_CHANNELS;
                        assert(depthChannelNr < DEPTH_CHANNELS);
                        cuv::ndarray_view<int, cuv::host_memory_space> channelView = depthImage[cuv::indices[depthChannelNr][cuv::index_range()][cuv::index_range()]];
                        calculateIntegral(channelView);
                    } else {
                        cuv::ndarray_view<float, cuv::host_memory_space> channelView = colorImage[cuv::indices[channelNr][cuv::index_range()][cuv::index_range()]];
                        calculateIntegral(channelView);
                    }
                }
            });

    integratedColor = true;
    integratedDepth = true;
}

void RGBDImage::calculateDerivative() {
    if (!integratedColor || !integratedDepth) {
        throw std::runtime_error("image not integrated");
    }

    tbb::parallel_for(tbb::blocked_range<size_t>(0, COLOR_CHANNELS + DEPTH_CHANNELS, 1),
            [&](const tbb::blocked_range<size_t>& range) {
                for(unsigned int channelNr = range.begin(); channelNr != range.end(); channelNr++) {
                    if (channelNr >= COLOR_CHANNELS) {
                        unsigned int depthChannelNr = channelNr - COLOR_CHANNELS;
                        assert(depthChannelNr < DEPTH_CHANNELS);
                        cuv::ndarray_view<int, cuv::host_memory_space> channelView = depthImage[cuv::indices[depthChannelNr][cuv::index_range()][cuv::index_range()]];
                        calculateDerivative(channelView);
                    } else {
                        cuv::ndarray_view<float, cuv::host_memory_space> channelView = colorImage[cuv::indices[channelNr][cuv::index_range()][cuv::index_range()]];
                        calculateDerivative(channelView);
                    }
                }
            });

    integratedColor = false;
    integratedDepth = false;
}

// http://www.cs.washington.edu/rgbd-dataset/trd5326jglrepxk649ed/rgbd-dataset_full/README.txt
void RGBDImage::saveDepth(const std::string& filename) const {

    cuv::ndarray<int, cuv::host_memory_space> tempDepthData = depthImage.copy();

    if (integratedDepth) {
        for (unsigned int c = 0; c < DEPTH_CHANNELS; c++) {
            cuv::ndarray_view<int, cuv::host_memory_space> channelView =
                    tempDepthData[cuv::indices[c][cuv::index_range()][cuv::index_range()]];
            calculateDerivative(channelView);
        }
    }

    vigra::UInt16Image uintImage(getWidth(), getHeight());
    for (int y = 0; y < getHeight(); ++y) {
        for (int x = 0; x < getWidth(); ++x) {
            int valid = tempDepthData(1, y, x);
            assert(valid == 0 || valid == 1);
            if (valid == 1) {
                uintImage(x, y) = tempDepthData(0, y, x);
            } else {
                uintImage(x, y) = 0;
            }
        }
    }

    vigra::exportImage(vigra::srcImageRange(uintImage),
            vigra::ImageExportInfo(filename.c_str()).setPixelType("UINT16"));
}

void RGBDImage::saveColor(const std::string& filename) const {

    vigra::DVector3Image image(getWidth(), getHeight());

    cuv::ndarray<float, cuv::host_memory_space> tmpColorImage = colorImage.copy();

    if (integratedColor) {
        for (unsigned int c = 0; c < COLOR_CHANNELS; ++c) {
            cuv::ndarray_view<float, cuv::host_memory_space> channelView =
                    tmpColorImage[cuv::indices[c][cuv::index_range()][cuv::index_range()]];
            calculateDerivative(channelView);
        }
    }

    for (unsigned int c = 0; c < COLOR_CHANNELS; ++c) {
        for (int y = 0; y < getHeight(); ++y) {
            for (int x = 0; x < getWidth(); ++x) {
                image(x, y)[c] = tmpColorImage(c, y, x);
            }
        }
    }

    if (inCIELab) {
        image = convertCIELab2RGB(image);
        vigra::exportImage(vigra::srcImageRange(image), vigra::ImageExportInfo(filename.c_str()));
    } else {
        vigra::UInt8RGBImage uintImage(getWidth(), getHeight());
        for (int y = 0; y < getHeight(); ++y) {
            for (int x = 0; x < getWidth(); ++x) {
                for (unsigned int c = 0; c < COLOR_CHANNELS; ++c) {
                    double value = image(x, y)[c];
                    assert(value >= 0.0 && value <= 255.0);
                    uintImage(x, y)[c] = value;
                }
            }
        }
        vigra::exportImage(vigra::srcImageRange(uintImage),
                vigra::ImageExportInfo(filename.c_str()).setPixelType("UINT8"));
    }

}

std::map<RGBColor, LabelType> colors;
tbb::mutex colorsMutex;

LabelType getOrAddColorId(const RGBColor& color, const LabelType& id) {
    if (colors.find(color) == colors.end()) {
        addColorId(color, id);
    }
    return colors[color];
}

void addColorId(const RGBColor& color, const LabelType& id) {
    if (colors.find(color) != colors.end()) {
        if (colors[color] != id) {
            std::ostringstream o;
            o << "color RGB(" << color << "): ";
            o << "existing label: " << static_cast<int>(colors[color]);
            o << ", new label: " << static_cast<int>(id);
            throw std::runtime_error(o.str());
        }
    } else {
        colors[color] = id;
    }
}

LabelType LabelImage::encodeColor(RGBColor color) {
	std::map<RGBColor, LabelType>::iterator it = colors.find(color);
	if (it != colors.end()) {
		return it->second;
	}

	LabelType id = static_cast<LabelType>(colors.size());
	addColorId(color, id);
	return id;
}

static LabelType getLabelType(const vigra::UInt8RGBImage& labelImage, int x, int y) {
    const vigra::RGBValue<vigra::UInt8> c = labelImage(x, y);
    RGBColor color(c[0], c[1], c[2]);

    return LabelImage::encodeColor(color);
}

RGBColor LabelImage::decodeLabel(const LabelType& v) {

    tbb::mutex::scoped_lock lock(colorsMutex);

    std::map<RGBColor, LabelType>::iterator it;
    for (it = colors.begin(); it != colors.end(); it++) {
        if (it->second == v) {
            lock.release();
            assert(it->first.size() == 3);
            return it->first;
        }
    }
    std::ostringstream o;
    o << "color for label " << static_cast<int>(v) << " not found" << std::endl;
    o << "available colors:" << std::endl;
    for (auto& c : colors) {
        o << static_cast<int>(c.second) << ": ";
        o << c.first[0] << ",";
        o << c.first[1] << ",";
        o << c.first[2];
        o << std::endl;
    }
    throw std::runtime_error(o.str());
}

LabelImage::LabelImage(const std::string& filename) :
        filename(filename) {

    vigra::UInt8RGBImage labelImage;
    try {
        loadImage(filename, labelImage);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("failed to load label image '") + filename + "': " + e.what());
    }

    width = labelImage.width();
    height = labelImage.height();
    image.resize(height, width);
    image = LabelType();

    utils::Timer timer;

    tbb::mutex::scoped_lock lock(colorsMutex);

    for (int x = 0; x < labelImage.width(); ++x) {
        for (int y = 0; y < labelImage.height(); ++y) {
            setLabel(x, y, getLabelType(labelImage, x, y));
        }
    }
}

void LabelImage::save(const std::string& filename) const {
    vigra::UInt8RGBImage labelImage(width, height);

    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            auto color = decodeLabel(getLabel(x, y));
            for (int c = 0; c < 3; c++) {
                labelImage(x, y)[c] = color[c];
            }
        }
    }

    vigra::exportImage(vigra::srcImageRange(labelImage),
            vigra::ImageExportInfo(filename.c_str()).setPixelType("UINT8"));
}

void LabelImage::resizeImage(int newWidth, int newHeight, LabelType paddingLabel)
{
	int originalWidth = getWidth();
	int originalHeight = getHeight();

	using namespace cuv;
	ndarray<LabelType, host_memory_space>  Li(extents[newHeight][newWidth]);

	Li = paddingLabel;

	Li[indices[index_range(0, originalHeight)][index_range(0, originalWidth)]] = image;

	image = Li;

	width = newWidth;
	height = newHeight;

}

LabeledRGBDImage::LabeledRGBDImage(const boost::shared_ptr<RGBDImage>& rgbdImage,
        const boost::shared_ptr<LabelImage>& labelImage) :
        rgbdImage(rgbdImage), labelImage(labelImage) {
    if (rgbdImage->getWidth() != labelImage->getWidth() || rgbdImage->getHeight() != labelImage->getHeight()) {
        std::ostringstream o;
        o << "RGB-D image '" << rgbdImage->getFilename();
        o << "' and label image '" << labelImage->getFilename();
        o << "' have different sizes: ";
        o << rgbdImage->getWidth() << "x" << rgbdImage->getHeight();
        o << " and " << labelImage->getWidth() << "x" << labelImage->getHeight();
        throw std::runtime_error(o.str());
    }
}

void LabeledRGBDImage::resizeImage(int newWidth, int newHeight, LabelType paddingLabel) const
{
	rgbdImage->resizeImage(newWidth, newHeight);
	labelImage->resizeImage(newWidth, newHeight, paddingLabel);
}

void LabeledRGBDImage::calculateIntegral() const
{
	rgbdImage->calculateIntegral();
}

LabeledRGBDImage loadImagePair(const std::string& filename, bool useCIELab,bool useDepthImages, bool useDepthFilling,
        bool calculateIntegralImages) {
    auto pos = filename.find("_colors.png");
    std::string labelFilename = filename;
    std::string depthFilename = filename;
    try {
        labelFilename.replace(pos, labelFilename.length(), "_ground_truth.png");
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("illegal label image filename: ") + labelFilename);
    }

	if (useDepthImages) {
		try {
			depthFilename.replace(pos, depthFilename.length(), "_depth.png");
		} catch (const std::exception& e) {
			throw std::runtime_error(std::string("illegal depth image filename: ") + depthFilename);
		}
	}

    const auto rgbdImage = boost::make_shared<RGBDImage>(filename, depthFilename, useDepthImages, useCIELab, useDepthFilling,
            calculateIntegralImages);
    const auto labelImage = boost::make_shared<LabelImage>(labelFilename);
    return LabeledRGBDImage(rgbdImage, labelImage);
}

std::vector<std::string> listImageFilenames(const std::string& path) {

    std::vector<std::string> filenames;

    fs::path targetDir(path);
    fs::directory_iterator eod;
    for (fs::directory_iterator it(targetDir); it != eod; it++) {
        const fs::path p = *it;
        if (fs::is_regular_file(p)) {
            std::string filename = p.native();
            if (filename.find("_colors.png") != std::string::npos) {
                filenames.push_back(filename);
            }
        } else if (fs::is_directory(p)) {
            for (const auto& filename : listImageFilenames(p.native())) {
                filenames.push_back(filename);
            }
        } else {
            throw std::runtime_error((boost::format("unknown path: %s") % p.native()).str());
        }
    }

    std::sort(filenames.begin(), filenames.end());

    return filenames;
}

LabelType getPaddingLabel(const std::vector<std::string>& ignoredColors) {
	RGBColor color;
	if (!ignoredColors.empty())
		color = RGBColor(ignoredColors[0]);
	else
		color = RGBColor(0,0,0);

	return LabelImage::encodeColor(color);
}

std::vector<LabeledRGBDImage> loadImages(const std::string& folder, bool useCIELab, bool useDepthImages, bool useDepthFilling,  const std::vector<std::string>& ignoredColors, size_t& numLabels) {

    std::vector<std::string> filenames = listImageFilenames(folder);
    CURFIL_INFO("going to load " << filenames.size() << " images from " << folder);

	//filenames.erase(filenames.begin()+100, filenames.end());

    size_t totalSizeInMemory = 0;
	int maxWidth = 0;
	int maxHeight = 0;

    LabeledRGBDImage emptyImage;
    std::vector<LabeledRGBDImage> images(filenames.size(), emptyImage);

	LabelType paddingLabel = getPaddingLabel(ignoredColors);

    tbb::mutex imageCounterMutex;
    size_t numImages = 0;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, images.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                for(size_t i = range.begin(); i != range.end(); i++) {

                    const auto& filename = filenames[i];
                    images[i] = loadImagePair(filename, useCIELab, useDepthImages, useDepthFilling);
                    {
                        tbb::mutex::scoped_lock lock(imageCounterMutex);
                        {
							if (images[i].getWidth() > maxWidth)
								maxWidth = images[i].getWidth();
							if (images[i].getHeight() > maxHeight)
								maxHeight = images[i].getHeight();
							if (++numImages % 50 == 0) {
                            CURFIL_INFO("loaded " << numImages << "/" << images.size() << " images");
							}
                        }
                    }

                }
            });


    bool firstResize = true;
	tbb::mutex firstresizeMutex;
	size_t imageSizeInMemory = 0;
	if (!images.empty()) {
		tbb::parallel_for(tbb::blocked_range<size_t>(0, images.size()),
				[&](const tbb::blocked_range<size_t>& range) {
					for(size_t i = range.begin(); i != range.end(); i++) {
						const LabeledRGBDImage& image = images[i];
						if (image.getWidth() != maxWidth || image.getHeight() != maxHeight)
						{
							if (firstResize){
								//added twice so that the lock is only performed the first time (1st),
								//and the message isn't displayed multiple times after a thread has acquired the lock (2nd)
								tbb::mutex::scoped_lock lock(firstresizeMutex);
								{
									if (firstResize)
									{	CURFIL_INFO("resizing images to " << maxWidth << "x" << maxHeight);
										firstResize = false;
									}
								}
							}
						image.resizeImage(maxWidth, maxHeight, paddingLabel);
						if (image.getWidth() != maxWidth || image.getHeight() != maxHeight) {
							std::ostringstream o;
							o << "Image " << image.getRGBDImage().getFilename()
							<< " has different size: ";
							o << image.getWidth() << "x" << image.getHeight();
							o << ". All images in the dataset should be resized to the maximum size("
							<< maxWidth << "x" << maxHeight << ")";
							throw std::runtime_error(o.str());
							}
						}
						if (imageSizeInMemory == 0)
						{
							imageSizeInMemory = image.getSizeInMemory();
						}
						totalSizeInMemory += image.getSizeInMemory();
			            if (image.getSizeInMemory() != imageSizeInMemory) {
			                std::ostringstream o;
			                o << "Image " << image.getRGBDImage().getFilename() << " has different size in memory: ";
			                o << image.getSizeInMemory() << " (expected: " << imageSizeInMemory << ").";
			                o << " This must not happen.";
			                throw std::runtime_error(o.str());
			            }
					}
				});

	}

    CURFIL_INFO("finished loading " << images.size() << " images. size in memory: "
            << (boost::format("%.2f MB") % (totalSizeInMemory / static_cast<double>(1024 * 1024))).str());

    numLabels = colors.size();

    return images;
}

}
