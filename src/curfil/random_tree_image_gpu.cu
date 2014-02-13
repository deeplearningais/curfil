#include "random_tree_image_gpu.h"

#include <boost/format.hpp>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <set>
#include <tbb/mutex.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "random_tree_image.h"
#include "score.h"
#include "utils.h"

namespace curfil {

// must be global
texture<float, cudaTextureType2DLayered, cudaReadModeElementType> colorTexture;
texture<int, cudaTextureType2DLayered, cudaReadModeElementType> depthTexture;

texture<float, cudaTextureType2DLayered, cudaReadModeElementType> treeTexture;

tbb::mutex initMutex;
volatile bool initialized = false;

tbb::mutex textureMutex;

const static int NUM_STREAMS = 2;
cudaStream_t streams[NUM_STREAMS] = { NULL, NULL };
ImageCache imageCache;
TreeCache treeCache;

__device__
float getColorChannelValue(int x, int y, int imageNr, int channel) {
    assert(channel < colorChannels);
    return tex2DLayered(colorTexture, x, y, imageNr * colorChannels + channel);
}

__device__
int getDepthValue(int x, int y, int imageNr) {
    return tex2DLayered(depthTexture, x, y, imageNr * depthChannels + depthChannel);
}

__device__
int getDepthValidValue(int x, int y, int imageNr) {
    return tex2DLayered(depthTexture, x, y, imageNr * depthChannels + depthValidChannel);
}

__device__
FeatureResponseType averageRegionDepth(int imageNr,
        const int16_t imageWidth, const int16_t imageHeight,
        int leftX, int rightX, int upperY, int lowerY) {

    if (leftX < 0 || rightX >= imageWidth || upperY < 0 || lowerY >= imageHeight) {
        return nan("");
    }

    int upperLeftValid = getDepthValidValue(leftX, upperY, imageNr);
    int upperRightValid = getDepthValidValue(rightX, upperY, imageNr);
    int lowerRightValid = getDepthValidValue(rightX, lowerY, imageNr);
    int lowerLeftValid = getDepthValidValue(leftX, lowerY, imageNr);

    int numValid = (lowerRightValid - upperRightValid) + (upperLeftValid - lowerLeftValid);
    assert(numValid >= 0 && numValid <= (rightX - leftX) * (lowerY - upperY));

    if (numValid == 0) {
        return nan("");
    }

    int upperLeftDepth = getDepthValue(leftX, upperY, imageNr);
    int upperRightDepth = getDepthValue(rightX, upperY, imageNr);
    int lowerRightDepth = getDepthValue(rightX, lowerY, imageNr);
    int lowerLeftDepth = getDepthValue(leftX, lowerY, imageNr);

    int sum = (lowerRightDepth - upperRightDepth) + (upperLeftDepth - lowerLeftDepth);
    FeatureResponseType feat = sum / static_cast<FeatureResponseType>(1000);
    return (feat / numValid);
}

__device__
FeatureResponseType averageRegionDepth(int imageNr,
        const int16_t imageWidth, const int16_t imageHeight,
        float depth,
        int sampleX, int sampleY,
        int offsetX, int offsetY,
        int regionWidth, int regionHeight) {

    int width = max(1, static_cast<int>(regionWidth / depth));
    int height = max(1, static_cast<int>(regionHeight / depth));
    int x = sampleX + static_cast<int>(offsetX / depth);
    int y = sampleY + static_cast<int>(offsetY / depth);

    int leftX = x - width;
    int rightX = x + width;
    int upperY = y - height;
    int lowerY = y + height;

    return averageRegionDepth(imageNr, imageWidth, imageHeight, leftX, rightX, upperY, lowerY);
}

__device__
FeatureResponseType averageRegionColor(int imageNr,
        uint16_t imageWidth, uint16_t imageHeight,
        int channel, float depth,
        int sampleX, int sampleY,
        int offsetX, int offsetY,
        int regionWidth, int regionHeight) {

    int width = max(1, static_cast<int>(regionWidth / depth));
    int height = max(1, static_cast<int>(regionHeight / depth));
    int x = sampleX + static_cast<int>(offsetX / depth);
    int y = sampleY + static_cast<int>(offsetY / depth);

    int leftX = x - width;
    int rightX = x + width;
    int upperY = y - height;
    int lowerY = y + height;

    if (leftX < 0 || rightX >= imageWidth || upperY < 0 || lowerY >= imageHeight) {
        return nan("");
    }

    FeatureResponseType upperLeftPixel = getColorChannelValue(leftX, upperY, imageNr, channel);
    FeatureResponseType upperRightPixel = getColorChannelValue(rightX, upperY, imageNr, channel);
    FeatureResponseType lowerRightPixel = getColorChannelValue(rightX, lowerY, imageNr, channel);
    FeatureResponseType lowerLeftPixel = getColorChannelValue(leftX, lowerY, imageNr, channel);

    if (isnan(lowerRightPixel) || isnan(lowerLeftPixel) || isnan(upperRightPixel) || isnan(upperLeftPixel))
    	  return nan("");

    FeatureResponseType sum = (lowerRightPixel - upperRightPixel) + (upperLeftPixel - lowerLeftPixel);

    return sum;
}

__device__
FeatureResponseType calculateDepthFeature(int imageNr,
        int16_t imageWidth, int16_t imageHeight,
        int8_t offset1X, int8_t offset1Y,
        int8_t offset2X, int8_t offset2Y,
        int8_t region1X, int8_t region1Y,
        int8_t region2X, int8_t region2Y,
        int sampleX, int sampleY, float depth) {

    FeatureResponseType a = averageRegionDepth(imageNr, imageWidth, imageHeight, depth, sampleX, sampleY, offset1X,
            offset1Y, region1X, region1Y);

    if (isnan(a))
        return a;

    FeatureResponseType b = averageRegionDepth(imageNr, imageWidth, imageHeight, depth, sampleX, sampleY, offset2X,
            offset2Y, region2X, region2Y);

    if (isnan(b))
        return b;

    return (a - b);
}

__device__
FeatureResponseType calculateColorFeature(int imageNr,
        const int16_t imageWidth, const int16_t imageHeight,
        int8_t offset1X, int8_t offset1Y,
        int8_t offset2X, int8_t offset2Y,
        int8_t region1X, int8_t region1Y,
        int8_t region2X, int8_t region2Y,
        int8_t channel1, int8_t channel2,
        int sampleX, int sampleY, float depth) {

    assert(channel1 >= 0 && channel1 < 3);
    assert(channel2 >= 0 && channel2 < 3);

    FeatureResponseType a = averageRegionColor(imageNr, imageWidth, imageHeight, channel1, depth, sampleX, sampleY,
            offset1X, offset1Y, region1X, region1Y);

    if (isnan(a))
        return a;

    FeatureResponseType b = averageRegionColor(imageNr, imageWidth, imageHeight, channel2, depth, sampleX, sampleY,
            offset2X, offset2Y, region2X, region2Y);

    if (isnan(b))
        return b;

    return (a - b);
}

__global__
void setupRandomStatesKernel(unsigned long long seed, curandState* state, unsigned int numFeatures) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < numFeatures) {
        /* Each thread gets same seed, a different sequence number, no offset */
        curand_init(seed, id, 0, &state[id]);
    }
}

__device__
void randomOffset(curandState* state, int8_t* x, int8_t* y, const uint8_t radius) {
    const uint8_t boxRadius = radius + 1;

    const int8_t vx = curand_uniform(state) * 2 * boxRadius - boxRadius;
    const int8_t vy = curand_uniform(state) * 2 * boxRadius - boxRadius;

    assert(vx >= -boxRadius && vx <= boxRadius);
    assert(vy >= -boxRadius && vy <= boxRadius);

    *x = vx;
    *y = vy;
}

__device__
void randomRegion(curandState* state, int8_t* x, int8_t* y, const uint8_t regionSize) {
    const int8_t vx = curand_uniform(state) * regionSize + 1;
    const int8_t vy = curand_uniform(state) * regionSize + 1;

    assert(vx >= 1 && vx <= regionSize);
    assert(vy >= 1 && vy <= regionSize);

    *x = vx;
    *y = vy;
}

__global__
void generateRandomFeaturesKernel(int seed,
        unsigned int numFeatures,
        int* keys, int* indices,
        uint16_t boxRadius,
        uint16_t regionSize,
        int8_t* types,
        int8_t* offsets1X, int8_t* offsets1Y,
        int8_t* regions1X, int8_t* regions1Y,
        int8_t* offsets2X, int8_t* offsets2Y,
        int8_t* regions2X, int8_t* regions2Y,
        int8_t* channels1, int8_t* channels2,
        float* thresholds,
        unsigned int numThresholds,
        unsigned int numSamples,
        int imageWidth, int imageHeight,
        int* imageNumbers,
        float* depths,
        int* sampleX,
        int* sampleY,
        uint8_t* sampleLabel,
        bool isUseDepthImages) {
    int feat = blockIdx.x * blockDim.x + threadIdx.x;

    if (feat >= numFeatures) {
        return;
    }

    curandState localState;
    curand_init(seed, feat, 0, &localState);

    uint8_t type;
    int8_t offset1X, offset1Y;
    int8_t offset2X, offset2Y;
    int8_t region1X, region1Y;
    int8_t region2X, region2Y;
    uint8_t channel1, channel2;

	if (isUseDepthImages)
		type = static_cast<uint8_t>(feat >= numFeatures / 2);
	else
		type = COLOR;

    types[feat] = type;

    randomOffset(&localState, &offset1X, &offset1Y, boxRadius);
    randomRegion(&localState, &region1X, &region1Y, regionSize);

    do {
        randomOffset(&localState, &offset2X, &offset2Y, boxRadius);
        randomRegion(&localState, &region2X, &region2Y, regionSize);
    } while (offset1X == offset2X && offset1Y == offset2Y);

    if (type == COLOR) {
        // chan1=MOD(INT(A2/(100/2)*3);3)
        // chan2=MOD(INT(A2/(100/2/3)*3);3)

    	if (isUseDepthImages) {
        channel1 = feat / (numFeatures / 2.0) * 3;
        channel1 %= 3;

        channel2 = feat / (numFeatures / 2.0 / 3) * 3;
        channel2 %= 3;
    	}
    	else
    	{
    	channel1 = feat / (numFeatures) * 3;
        channel1 %= 3;

        channel2 = feat / (numFeatures / 3) * 3;
        channel2 %= 3;
    	}

        // channel1 = curand_uniform(&localState) * 3;
        // channel2 = curand_uniform(&localState) * 3;
    } else {
        channel1 = 0;
        channel2 = 0;
    }

    for (unsigned int thresh = 0; thresh < numThresholds; thresh++) {
        unsigned int numSample = curand_uniform(&localState) * (numSamples - 1);

        FeatureResponseType featureResponse;
        switch (type) {
            case COLOR:
                featureResponse = calculateColorFeature(imageNumbers[numSample],
                        imageWidth, imageHeight,
                        offset1X, offset1Y,
                        offset2X, offset2Y,
                        region1X, region1Y,
                        region2X, region2Y,
                        channel1, channel2,
                        sampleX[numSample], sampleY[numSample], depths[numSample]);
                break;
            case DEPTH:
                featureResponse = calculateDepthFeature(imageNumbers[numSample],
                        imageWidth, imageHeight,
                        offset1X, offset1Y,
                        offset2X, offset2Y,
                        region1X, region1Y,
                        region2X, region2Y,
                        sampleX[numSample], sampleY[numSample], depths[numSample]);
                break;
            default:
                assert(false);
                break;
        }

        if (isnan(featureResponse)) {
            featureResponse = 0.0;
        }

        thresholds[thresh * numFeatures + feat] = featureResponse;
    }

    int32_t sortKey = 0;
    sortKey |= static_cast<uint8_t>(type & 0x03) << 30; // 2 bit for the type
    sortKey |= static_cast<uint8_t>(channel1 & 0x0F) << 26; // 4 bit for channel1
    sortKey |= static_cast<uint8_t>(channel2 & 0x0F) << 22; // 4 bit for channel2
    sortKey |= static_cast<uint8_t>((offset1Y + 127) & 0xFF) << 14; // 8 bit for offset1.y
    sortKey |= static_cast<uint8_t>((offset1X + 127) & 0xFF) << 6; // 8 bit for offset1.x

    keys[feat] = sortKey;

    assert(keys[feat] >= 0);
    indices[feat] = feat;

    offsets1X[feat] = offset1X;
    offsets1Y[feat] = offset1Y;
    regions1X[feat] = region1X;
    regions1Y[feat] = region1Y;

    offsets2X[feat] = offset2X;
    offsets2Y[feat] = offset2Y;
    regions2X[feat] = region2X;
    regions2Y[feat] = region2Y;

    channels1[feat] = channel1;
    channels2[feat] = channel2;
}

Samples<cuv::dev_memory_space> ImageFeatureEvaluation::copySamplesToDevice(
        const std::vector<const PixelInstance*>& samples, cudaStream_t stream) {

    imageCache.copyImages(configuration.getImageCacheSize(), samples);
    cudaSafeCall(cudaStreamSynchronize(stream));

    utils::Profile p("copySamplesToDevice");

    Samples<cuv::host_memory_space> samplesOnHost(samples.size(), sampleDataAllocator);

    for (size_t i = 0; i < samples.size(); i++) {
        const PixelInstance* sample = samples[i];
        samplesOnHost.imageNumbers[i] = imageCache.getElementPos(sample->getRGBDImage());
        samplesOnHost.depths[i] = sample->getDepth().getFloatValue();
        samplesOnHost.sampleX[i] = sample->getX();
        samplesOnHost.sampleY[i] = sample->getY();
        samplesOnHost.labels[i] = sample->getLabel();
        samplesOnHost.useFlipping[i] = sample->getFlipping();
    }

    utils::Timer copySamplesAssignTimer;

    Samples<cuv::dev_memory_space> samplesOnDevice(samplesOnHost, stream);
    cudaSafeCall(cudaStreamSynchronize(stream));
    return samplesOnDevice;
}

void clearImageCache() {
    CURFIL_INFO("clearing image cache");
    imageCache.clear();
}

TreeNodes::TreeNodes(const TreeNodes& other) :
        m_treeId(other.getTreeId()),
                m_numNodes(other.numNodes()),
                m_numLabels(other.numLabels()),
                m_sizePerNode(other.sizePerNode()),
                m_data(other.data())
{
}

TreeNodes::TreeNodes(const boost::shared_ptr<const RandomTree<PixelInstance, ImageFeatureFunction> >& tree) :
        m_treeId(tree->getTreeId()),
                m_numNodes(tree->countNodes()),
                m_numLabels(tree->getNumClasses()),
                m_sizePerNode(offsetHistograms + sizeof(float) * m_numLabels),
                m_data(LAYERS_PER_TREE * NODES_PER_TREE_LAYER, m_sizePerNode)
{
    assert(offsetHistograms == 24);

    const unsigned int MAX_NODES = LAYERS_PER_TREE * NODES_PER_TREE_LAYER;
    if (m_numNodes > MAX_NODES) {
        throw std::runtime_error((boost::format("too many nodes in tree %d: %d (max: %d)")
                % tree->getTreeId() % m_numNodes % MAX_NODES).str());
    }
    convert(tree);
    assert(m_numLabels == tree->getHistogram().size());
}

template<class T>
void TreeNodes::setValue(size_t node, size_t offset, const T& value) {

    const size_t layer = node / NODES_PER_TREE_LAYER;
    const size_t nodeOffset = node % NODES_PER_TREE_LAYER;

    assert(layer * NODES_PER_TREE_LAYER + nodeOffset == node);

    if (layer >= LAYERS_PER_TREE) {
        throw std::runtime_error((boost::format("illegal layer: %d (node: %d)")
                % layer % node).str());
    }

    T* ptr = reinterpret_cast<T*>(m_data.ptr()
            + layer * NODES_PER_TREE_LAYER * m_sizePerNode
            + nodeOffset * m_sizePerNode + offset);

    // CURFIL_DEBUG("setting value for node " << node << " at pos (" << layer << "," << nodeOffset << "," << offset << ") to " << static_cast<double>(value));

    *ptr = value;
}

void TreeNodes::setLeftNodeOffset(size_t node, int offset) {
    setValue(node, offsetLeftNode, offset);
}

void TreeNodes::setThreshold(size_t node, float threshold) {
    setValue(node, offsetThreshold, threshold);
}

void TreeNodes::setHistogramValue(size_t node, size_t label, float value) {
    assert(offsetHistograms == 6 * sizeof(float));
    setValue(node, offsetHistograms + label * sizeof(value), value);
}

void TreeNodes::setType(size_t node, int8_t value) {
    setValue(node, offsetTypes, static_cast<int>(value));
}

void TreeNodes::setOffset1X(size_t node, int8_t value) {
    setValue(node, offsetFeatures + 0 * sizeof(value), value);
}

void TreeNodes::setOffset1Y(size_t node, int8_t value) {
    setValue(node, offsetFeatures + 1 * sizeof(value), value);
}

void TreeNodes::setRegion1X(size_t node, int8_t value) {
    setValue(node, offsetFeatures + 2 * sizeof(value), value);
}

void TreeNodes::setRegion1Y(size_t node, int8_t value) {
    setValue(node, offsetFeatures + 3 * sizeof(value), value);
}

void TreeNodes::setOffset2X(size_t node, int8_t value) {
    setValue(node, offsetFeatures + 4 * sizeof(value), value);
}

void TreeNodes::setOffset2Y(size_t node, int8_t value) {
    setValue(node, offsetFeatures + 5 * sizeof(value), value);
}

void TreeNodes::setRegion2X(size_t node, int8_t value) {
    setValue(node, offsetFeatures + 6 * sizeof(value), value);
}

void TreeNodes::setRegion2Y(size_t node, int8_t value) {
    setValue(node, offsetFeatures + 7 * sizeof(value), value);
}

void TreeNodes::setChannel1(size_t node, uint16_t value) {
    setValue(node, offsetChannels + 0 * sizeof(value), value);
}

void TreeNodes::setChannel2(size_t node, uint16_t value) {
    setValue(node, offsetChannels + 1 * sizeof(value), value);
}

void TreeNodes::convert(const boost::shared_ptr<const RandomTree<PixelInstance, ImageFeatureFunction> >& tree) {

    size_t offset = tree->getNodeId() - tree->getTreeId();

    if (offset >= m_numNodes) {
        throw std::runtime_error((boost::format("tree %d, illegal offset: %d (numNodes: %d)")
                % tree->getTreeId() % offset % m_numNodes).str());
    }

    // could be limited to the leaf-node case
    const cuv::ndarray<double, cuv::host_memory_space>& histogram = tree->getNormalizedHistogram();
    assert(histogram.ndim() == 1);
    assert(histogram.shape(0) == m_numLabels);
    for (size_t label = 0; label < histogram.shape(0); label++) {
        setHistogramValue(offset, label, static_cast<float>(histogram(label)));
    }

    if (tree->isLeaf()) {
        setLeftNodeOffset(offset, -1);
        setThreshold(offset, std::numeric_limits<float>::quiet_NaN());
        return;
    }

    // decision node
    const ImageFeatureFunction& feature = tree->getSplit().getFeature();
    setType(offset, static_cast<int8_t>(feature.getType()));
    setOffset1X(offset, static_cast<int8_t>(feature.getOffset1().getX()));
    setOffset1Y(offset, static_cast<int8_t>(feature.getOffset1().getY()));
    setRegion1X(offset, static_cast<int8_t>(feature.getRegion1().getX()));
    setRegion1Y(offset, static_cast<int8_t>(feature.getRegion1().getY()));
    setOffset2X(offset, static_cast<int8_t>(feature.getOffset2().getX()));
    setOffset2Y(offset, static_cast<int8_t>(feature.getOffset2().getY()));
    setRegion2X(offset, static_cast<int8_t>(feature.getRegion2().getX()));
    setRegion2Y(offset, static_cast<int8_t>(feature.getRegion2().getY()));
    setChannel1(offset, static_cast<int8_t>(feature.getChannel1()));
    setChannel2(offset, static_cast<int8_t>(feature.getChannel2()));

    setThreshold(offset, tree->getSplit().getThreshold());

    convert(tree->getLeft());
    convert(tree->getRight());

    // tree nodes must be already in breadth-first order
    assert(tree->getRight()->getNodeId() == tree->getLeft()->getNodeId() + 1);

    const int leftNodeOffset = tree->getLeft()->getNodeId() - tree->getNodeId();
    assert(leftNodeOffset > 0);
    setLeftNodeOffset(offset, leftNodeOffset);
}

void DeviceCache::setBound(bool bound) {
    assert(bound != this->bound);
    this->bound = bound;
}

DeviceCache::~DeviceCache() {
    // clear() must be called in destructor of derived class
    assert(!bound);
    assert(elementTimes.empty());
    assert(elementIdMap.empty());
    assert(currentTime == 0);
}

bool DeviceCache::containsElement(const void* element) const {
    assert(element);
    return (elementIdMap.find(element) != elementIdMap.end());
}

size_t DeviceCache::getElementPos(const void* element) const {
    assert(element);
    std::map<const void*, size_t>::const_iterator it = elementIdMap.find(element);
    if (it == elementIdMap.end()) {
        throw std::runtime_error(getElementName(element) + " not found in cache");
    }
    return it->second;
}

void DeviceCache::clear() {
    if (bound) {
        unbind();
    }

    freeArray();

    elementTimes.clear();
    elementIdMap.clear();
    currentTime = 0;
}

void DeviceCache::copyElements(size_t cacheSize, const std::set<const void*>& elements) {

    if (elements.empty())
        return;

    if (elements.size() > cacheSize) {
        throw std::runtime_error(boost::str(boost::format("too many images: %d. max: %d")
                % elements.size()
                % cacheSize));
    }

    if (cacheSize != this->cacheSize) {
        clear();
        this->cacheSize = cacheSize;
    }

    currentTime++;

    size_t numTransferred = 0;

    const boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();

    std::set<const void*>::const_iterator it;
    for (it = elements.begin(); it != elements.end(); it++) {

        const void* element = *it;

        size_t elementPos = 0;

        // check if image is already there
        if (elementIdMap.find(element) != elementIdMap.end()) {
            elementPos = elementIdMap[element];
            // update LRU vector time
            elementTimes[elementPos] = currentTime;
            CURFIL_DEBUG(getElementName(element) << " already in device cache");
            continue;
        }

        // element does not exist yet. transfer it

        CURFIL_DEBUG(getElementName(element) << " not yet on device. transferring");

        if (elementIdMap.size() < cacheSize) {
            elementPos = elementIdMap.size();
        } else {

            // find least recently used element
            size_t oldestTime = currentTime;
            std::map<size_t, size_t>::const_iterator it;
            for (it = elementTimes.begin(); it != elementTimes.end(); it++) {
                if (it->second < oldestTime) {
                    oldestTime = it->second;
                    elementPos = it->first;
                }
            }

            assert(oldestTime < currentTime);

            CURFIL_DEBUG("replacing " << getElementName(element)
                    << " (time: " << oldestTime << ", current: " << currentTime << ")");

            {
                std::map<const void*, size_t>::iterator it;
                for (it = elementIdMap.begin(); it != elementIdMap.end(); it++) {
                    if (it->second == elementPos) {
                        CURFIL_DEBUG("removing " << getElementName(it->first) << " at pos " << elementPos);
                        elementIdMap.erase(it);
                        break;
                    }
                }
            }

        }

        elementIdMap[element] = elementPos;
        elementTimes[elementPos] = currentTime;

        CURFIL_DEBUG("transfer " << getElementName(element) << " to pos " << elementPos);

        if (bound) {
            unbind();
        }

        transferElement(elementPos, element, streams[0]);
        numTransferred++;
    }

    if (numTransferred > 0) {
        CURFIL_DEBUG("transferred " << numTransferred << "/" << elements.size() << " "
                << getElementsName() << " from host to device");

        cudaSafeCall(cudaStreamSynchronize(streams[0]));
        const boost::posix_time::ptime stop = boost::posix_time::microsec_clock::local_time();
        totalTransferTimeMicroseconds += (stop - start).total_microseconds();
    }

    if (!bound) {
        bind();
    }
}

void DeviceCache::updateCacheSize(size_t cacheSize) {
    if (cacheSize != this->cacheSize) {
        clear();
        this->cacheSize = cacheSize;
    }
}

ImageCache::~ImageCache() {
    CURFIL_DEBUG("destroying image cache " << this);
    clear();
}

void ImageCache::freeArray() {

    assert(!isBound());

    if (colorTextureData != NULL) {
        cudaFreeArray(colorTextureData);
        colorTextureData = NULL;
    }

    if (depthTextureData != NULL) {
        cudaFreeArray(depthTextureData);
        depthTextureData = NULL;
    }
}

void ImageCache::allocArray() {

    assert(!isBound());

    unsigned int flags = cudaArrayLayered;

    assert(colorTextureData == NULL);
    assert(depthTextureData == NULL);

    assert(this->width > 0);
    assert(this->height > 0);
    assert(getCacheSize() > 0);

    {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaExtent extent = make_cudaExtent(width, height, colorChannels * getCacheSize());
        cudaSafeCall(cudaMalloc3DArray(&colorTextureData, &channelDesc, extent, flags));
    }

    {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
        cudaExtent extent = make_cudaExtent(width, height, depthChannels * getCacheSize());
        cudaSafeCall(cudaMalloc3DArray(&depthTextureData, &channelDesc, extent, flags));
    }
}

ImageCache::ImageCache() :
        DeviceCache(), width(0), height(0), colorTextureData(NULL), depthTextureData(NULL) {
}

void ImageCache::copyImages(size_t cacheSize, const std::vector<const PixelInstance*>& samples) {

    std::set<const RGBDImage*> images;
    for (size_t sample = 0; sample < samples.size(); sample++) {
        const RGBDImage* image = samples[sample]->getRGBDImage();
        images.insert(image);
    }

    copyImages(cacheSize, images);
}

void ImageCache::copyImages(size_t cacheSize, const std::set<const RGBDImage*>& images) {

    if (images.empty())
        return;

    int width = (*images.begin())->getWidth();
    int height = (*images.begin())->getHeight();

#ifndef NDEBUG
    {
        std::set<const RGBDImage*>::const_iterator it;
        for (it = images.begin(); it != images.end(); it++) {
            assert(width == (*it)->getWidth());
            assert(height == (*it)->getHeight());
        }
    }
#endif

    if (width != this->width || height != this->height) {
        this->width = width;
        this->height = height;
        clear();
    }

    updateCacheSize(cacheSize);

    if (colorTextureData == NULL) {
        allocArray();
    }

    std::set<const void*> elements;
    std::set<const RGBDImage*>::const_iterator it;
    for (it = images.begin(); it != images.end(); it++) {
        elements.insert(*it);
    }

    copyElements(cacheSize, elements);
}

void ImageCache::transferElement(size_t imagePos, const void* imagePtr, cudaStream_t stream) {

    const RGBDImage* image = reinterpret_cast<const RGBDImage*>(imagePtr);

    struct cudaMemcpy3DParms colorCopyParams;
    memset(&colorCopyParams, 0, sizeof(colorCopyParams));
    colorCopyParams.extent = make_cudaExtent(width, height, colorChannels);
    colorCopyParams.kind = cudaMemcpyHostToDevice;
    colorCopyParams.dstArray = colorTextureData;

    struct cudaMemcpy3DParms depthCopyParams;
    memset(&depthCopyParams, 0, sizeof(depthCopyParams));
    depthCopyParams.extent = make_cudaExtent(width, height, depthChannels);
    depthCopyParams.kind = cudaMemcpyHostToDevice;
    depthCopyParams.dstArray = depthTextureData;

    assert(image->getColorImage().ndim() == 3);
    assert(image->getColorImage().shape(0) == static_cast<unsigned int>(colorChannels));
    assert(image->getColorImage().shape(1) == static_cast<unsigned int>(height));
    assert(image->getColorImage().shape(2) == static_cast<unsigned int>(width));
    colorCopyParams.dstPos = make_cudaPos(0, 0, colorChannels * imagePos);
    colorCopyParams.srcPtr = make_cudaPitchedPtr(
            const_cast<void*>(reinterpret_cast<const void*>(image->getColorImage().ptr())),
            sizeof(float) * width, width, height);
    cudaSafeCall(cudaMemcpy3DAsync(&colorCopyParams, stream));

    assert(image->getDepthImage().ndim() == 3);
    assert(image->getDepthImage().shape(0) == static_cast<unsigned int>(depthChannels));
    assert(image->getDepthImage().shape(1) == static_cast<unsigned int>(height));
    assert(image->getDepthImage().shape(2) == static_cast<unsigned int>(width));
    depthCopyParams.dstPos = make_cudaPos(0, 0, depthChannels * imagePos);
    depthCopyParams.srcPtr = make_cudaPitchedPtr(
            const_cast<void*>(reinterpret_cast<const void*>(image->getDepthImage().ptr())),
            sizeof(int) * width, width, height);
    cudaSafeCall(cudaMemcpy3DAsync(&depthCopyParams, stream));
}

std::string ImageCache::getElementName(const void* imagePtr) const {
    const RGBDImage* image = reinterpret_cast<const RGBDImage*>(imagePtr);
    return (boost::format("image %p") % image).str();
}

std::string ImageCache::getElementsName() const {
    return "images";
}

void ImageCache::bind() {

    assert(!isBound());

    colorTexture.normalized = false;
    colorTexture.filterMode = cudaFilterModePoint;
    colorTexture.addressMode[0] = cudaAddressModeClamp;
    colorTexture.addressMode[1] = cudaAddressModeClamp;
    colorTexture.addressMode[2] = cudaAddressModeClamp;

    assert(colorTextureData != NULL);

    cudaSafeCall(cudaBindTextureToArray(colorTexture, colorTextureData));

    depthTexture.normalized = false;
    depthTexture.filterMode = cudaFilterModePoint;
    depthTexture.addressMode[0] = cudaAddressModeClamp;
    depthTexture.addressMode[1] = cudaAddressModeClamp;
    depthTexture.addressMode[2] = cudaAddressModeClamp;

    assert(depthTextureData != NULL);

    cudaSafeCall(cudaBindTextureToArray(depthTexture, depthTextureData));

    setBound(true);
}

void ImageCache::unbind() {

    assert(isBound());

    cudaUnbindTexture(colorTexture);
    cudaUnbindTexture(depthTexture);

    setBound(false);
}

TreeCache::~TreeCache() {
    CURFIL_DEBUG("destroying tree cache " << this);
    clear();
}

void TreeCache::freeArray() {

    assert(!isBound());

    if (treeTextureData != NULL) {
        cudaFreeArray(treeTextureData);
        treeTextureData = NULL;
    }
}

void TreeCache::allocArray() {

    assert(!isBound());

    assert(treeTextureData == NULL);

    assert(sizePerNode > 0);
    assert(numLabels > 0);
    assert(getCacheSize() > 0);

    CURFIL_INFO("tree cache: allocating " << getCacheSize() << " x " << LAYERS_PER_TREE << " x "
            << NODES_PER_TREE_LAYER << " x " << sizePerNode << " bytes");

    {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaExtent extent = make_cudaExtent(sizePerNode / sizeof(float), NODES_PER_TREE_LAYER,
                LAYERS_PER_TREE * getCacheSize());
        cudaSafeCall(cudaMalloc3DArray(&treeTextureData, &channelDesc, extent, cudaArrayLayered));
    }
}

TreeCache::TreeCache() :
        DeviceCache(), sizePerNode(0), numLabels(0),
                treeTextureData(NULL) {
}

void TreeCache::copyTree(size_t cacheSize, const TreeNodes* tree) {
    std::set<const TreeNodes*> trees;
    trees.insert(tree);
    copyTrees(cacheSize, trees);
}

void TreeCache::copyTrees(size_t cacheSize, const std::set<const TreeNodes*>& trees) {

    if (trees.empty())
        return;

    const size_t sizePerNode = (*trees.begin())->sizePerNode();
    const LabelType numLabels = (*trees.begin())->numLabels();

    {
        std::set<const TreeNodes*>::const_iterator it;
        for (it = trees.begin(); it != trees.end(); it++) {
            assert(sizePerNode == (*it)->sizePerNode());
            assert(numLabels == (*it)->numLabels());
        }
    }

    if (numLabels != this->numLabels || sizePerNode != this->sizePerNode) {
        this->numLabels = numLabels;
        this->sizePerNode = sizePerNode;
        clear();
    }

    updateCacheSize(cacheSize);

    if (treeTextureData == NULL) {
        allocArray();
    }

    std::set<const void*> elements;
    std::set<const TreeNodes*>::const_iterator it;
    for (it = trees.begin(); it != trees.end(); it++) {
        elements.insert(*it);
    }
    copyElements(cacheSize, elements);
}

void TreeCache::transferElement(size_t elementPos, const void* element, cudaStream_t stream) {

    assert(!isBound());

    utils::Profile profile("transferTree");

    const TreeNodes* tree = reinterpret_cast<const TreeNodes*>(element);

    struct cudaMemcpy3DParms copyParams;
    memset(&copyParams, 0, sizeof(copyParams));
    copyParams.kind = cudaMemcpyHostToDevice;
    copyParams.dstArray = treeTextureData;

    assert(elementPos < getCacheSize());

    const size_t layers = ceil(tree->numNodes() / static_cast<double>(NODES_PER_TREE_LAYER));
    assert(layers >= 1);
    assert(layers <= LAYERS_PER_TREE);

    copyParams.dstPos = make_cudaPos(0, 0, elementPos * LAYERS_PER_TREE);
    copyParams.extent = make_cudaExtent(sizePerNode / sizeof(float), NODES_PER_TREE_LAYER, layers);
    void* ptr = const_cast<void*>(reinterpret_cast<const void*>(tree->data().ptr()));

    CURFIL_INFO("transfer " << getElementName(element) << " to pos " << elementPos
            << " (layer " << elementPos * LAYERS_PER_TREE << ")"
            << " with " << tree->numNodes() << " nodes in " << layers << " layers");

    assert(tree->data().size() == sizePerNode * NODES_PER_TREE_LAYER * LAYERS_PER_TREE);

    copyParams.srcPtr = make_cudaPitchedPtr(ptr, sizePerNode, sizePerNode / sizeof(float), NODES_PER_TREE_LAYER);
    cudaSafeCall(cudaMemcpy3DAsync(&copyParams, stream));
}

std::string TreeCache::getElementName(const void* element) const {
    const TreeNodes* tree = reinterpret_cast<const TreeNodes*>(element);
    return (boost::format("tree %d (%p)") % tree->getTreeId() % tree).str();
}

std::string TreeCache::getElementsName() const {
    return "trees";
}

void TreeCache::bind() {

    assert(!isBound());

    treeTexture.normalized = false;
    treeTexture.filterMode = cudaFilterModePoint;
    treeTexture.addressMode[0] = cudaAddressModeClamp;
    treeTexture.addressMode[1] = cudaAddressModeClamp;
    treeTexture.addressMode[2] = cudaAddressModeClamp;

    assert(treeTextureData != NULL);

    cudaSafeCall(cudaBindTextureToArray(treeTexture, treeTextureData));

    setBound(true);
}

void TreeCache::unbind() {

    assert(isBound());

    cudaUnbindTexture(treeTexture);

    setBound(false);
}

__device__
static size_t featureResponseOffset(size_t sample, size_t feature,
        size_t numSamples, size_t numFeatures) {
    // XXX: also need to change pointer arithmetic in aggregateHistogramsKernel
    // return sample * numFeatures + feature;
    return feature * numSamples + sample;
}

__device__
static unsigned int counterOffset(unsigned int label, unsigned int value, unsigned int threshold, unsigned int feature,
        unsigned int numLabels, unsigned int numFeatures, unsigned int numThresholds) {
    assert(value == 0 || value == 1);

    assert(label < numLabels);
    assert(feature < numFeatures);
    assert(threshold < numThresholds);

    // size_t index = (2 * label + value) * numFeatures * numThresholds + threshold * numFeatures + feature;
    // features × thresholds × labels × 2
    unsigned int index = feature * numThresholds * numLabels * 2;
    index += threshold * numLabels * 2;
    index += label * 2;
    index += value;
    return index;
}

__device__
int getNodeOffset(int node, int tree) {
    return node % NODES_PER_TREE_LAYER;
}

__device__
int getLayer(int node, int tree) {
    return tree * LAYERS_PER_TREE + node / NODES_PER_TREE_LAYER;
}

__device__
int getLeftNodeOffset(int node, int tree) {
    float v = tex2DLayered(treeTexture, 0, getNodeOffset(node, tree), getLayer(node, tree));
    return (*reinterpret_cast<int*>(&v));
}

__device__
int getType(int node, int tree) {
    float v = tex2DLayered(treeTexture, 1, getNodeOffset(node, tree), getLayer(node, tree));
    return (*reinterpret_cast<int*>(&v));
}

__device__
char4 getParam1(int node, int tree) {
    float v = tex2DLayered(treeTexture, 2, getNodeOffset(node, tree), getLayer(node, tree));
    return (*reinterpret_cast<char4*>(&v));
}

__device__
char4 getParam2(int node, int tree) {
    float v = tex2DLayered(treeTexture, 3, getNodeOffset(node, tree), getLayer(node, tree));
    return (*reinterpret_cast<char4*>(&v));
}

__device__
ushort2 getChannels(int node, int tree) {
    float v = tex2DLayered(treeTexture, 4, getNodeOffset(node, tree), getLayer(node, tree));
    return (*reinterpret_cast<ushort2*>(&v));
}

__device__
float getThreshold(int node, int tree) {
    return tex2DLayered(treeTexture, 5, getNodeOffset(node, tree), getLayer(node, tree));
}

__device__
float getHistogramValue(int label, int node, int tree) {
    return tex2DLayered(treeTexture, 6 + label, getNodeOffset(node, tree), getLayer(node, tree));
}

// for the unit test
__global__ void fetchTreeNodeData(
        int* leftNodeOffset,
        int* type,
        int8_t* offset1X, int8_t* offset1Y,
        int8_t* region1X, int8_t* region1Y,
        int8_t* offset2X, int8_t* offset2Y,
        int8_t* region2X, int8_t* region2Y,
        int8_t* channel1, int8_t* channel2,
        float* threshold,
        float* histogram,
        const int node, const int tree, const int numLabels) {

    assert(threadIdx.x == 0);
    assert(blockDim.y == 1);
    assert(blockDim.x == 1);

    *leftNodeOffset = getLeftNodeOffset(node, tree);
    *type = getType(node, tree);
    char4 param1 = getParam1(node, tree);
    *offset1X = param1.x;
    *offset1Y = param1.y;
    *region1X = param1.z;
    *region1Y = param1.w;

    char4 param2 = getParam2(node, tree);
    *offset2X = param2.x;
    *offset2Y = param2.y;
    *region2X = param2.z;
    *region2Y = param2.w;

    ushort2 channels = getChannels(node, tree);
    *channel1 = channels.x;
    *channel2 = channels.y;

    *threshold = getThreshold(node, tree);

    for (int label = 0; label < numLabels; label++) {
        histogram[label] = getHistogramValue(label, node, tree);
    }
}

// for the unit test
TreeNodeData getTreeNode(const int nodeNr, const boost::shared_ptr<const TreeNodes>& treeData) {
    treeCache.copyTree(3, treeData.get());

    const size_t nodeOffset = nodeNr - treeData->getTreeId();

    const size_t numLabels = treeData->numLabels();

    cuv::ndarray<int, cuv::dev_memory_space> leftNodeOffset(1);
    cuv::ndarray<int, cuv::dev_memory_space> type(1);
    cuv::ndarray<int8_t, cuv::dev_memory_space> offset1X(1);
    cuv::ndarray<int8_t, cuv::dev_memory_space> offset1Y(1);
    cuv::ndarray<int8_t, cuv::dev_memory_space> region1X(1);
    cuv::ndarray<int8_t, cuv::dev_memory_space> region1Y(1);
    cuv::ndarray<int8_t, cuv::dev_memory_space> offset2X(1);
    cuv::ndarray<int8_t, cuv::dev_memory_space> offset2Y(1);
    cuv::ndarray<int8_t, cuv::dev_memory_space> region2X(1);
    cuv::ndarray<int8_t, cuv::dev_memory_space> region2Y(1);
    cuv::ndarray<int8_t, cuv::dev_memory_space> channel1(1);
    cuv::ndarray<int8_t, cuv::dev_memory_space> channel2(1);
    cuv::ndarray<float, cuv::dev_memory_space> threshold(1);
    cuv::ndarray<float, cuv::dev_memory_space> histogram(numLabels);

    int treeNr = treeCache.getElementPos(treeData.get());

    fetchTreeNodeData<<<1,1>>>(leftNodeOffset.ptr(),
            type.ptr(),
            offset1X.ptr(), offset1Y.ptr(),
            region1X.ptr(), region1Y.ptr(),
            offset2X.ptr(), offset2Y.ptr(),
            region2X.ptr(), region2Y.ptr(),
            channel1.ptr(), channel2.ptr(),
            threshold.ptr(), histogram.ptr(),
            nodeOffset, treeNr, numLabels);
    cudaSafeCall(cudaDeviceSynchronize());

    TreeNodeData data;
    data.leftNodeOffset = leftNodeOffset[0];
    data.type = type[0];
    data.offset1X = offset1X[0];
    data.offset1Y = offset1Y[0];
    data.region1X = region1X[0];
    data.region1Y = region1Y[0];
    data.offset2X = offset2X[0];
    data.offset2Y = offset2Y[0];
    data.region2X = region2X[0];
    data.region2Y = region2Y[0];
    data.channel1 = channel1[0];
    data.channel2 = channel2[0];
    data.threshold = threshold[0];
    data.histogram = cuv::ndarray<float, cuv::host_memory_space>(numLabels);
    for (size_t label = 0; label < numLabels; label++) {
        data.histogram[label] = histogram(label);
    }
    return data;

}

__global__ void classifyKernel(
        float* output, int tree,
        const int16_t imageWidth, const int16_t imageHeight,
        const LabelType numLabels, bool useDepthImages) {

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= imageWidth) {
        return;
    }

    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= imageHeight) {
        return;
    }

    float depth;

    // depth might be nan here
    if (useDepthImages)
    	depth = averageRegionDepth(0, imageWidth, imageHeight, x, x + 1, y, y + 1);
    else
    	depth = 1;

    int currentNodeOffset = 0;
    while (true) {
        const int16_t leftNodeOffset = getLeftNodeOffset(currentNodeOffset, tree);
        assert(leftNodeOffset == -1 || leftNodeOffset > 0);
        if (leftNodeOffset < 0) {
            assert(isnan(getThreshold(currentNodeOffset, tree)));
            for (LabelType label = 0; label < numLabels; label++) {
                float v = getHistogramValue(label, currentNodeOffset, tree);
                assert(!isnan(v));
                assert(v >= 0.0);
                output[label * imageWidth * imageHeight + y * imageWidth + x] += v;
            }
            // leaf node
            return;
        }

        char4 param1 = getParam1(currentNodeOffset, tree);
        int8_t offset1X = param1.x;
        int8_t offset1Y = param1.y;
        int8_t region1X = param1.z;
        int8_t region1Y = param1.w;

        char4 param2 = getParam2(currentNodeOffset, tree);
        int8_t offset2X = param2.x;
        int8_t offset2Y = param2.y;
        int8_t region2X = param2.z;
        int8_t region2Y = param2.w;

        FeatureResponseType featureResponse;
        switch (getType(currentNodeOffset, tree)) {
            case COLOR: {
                ushort2 channels = getChannels(currentNodeOffset, tree);
                featureResponse = calculateColorFeature(0,
                        imageWidth, imageHeight,
                        offset1X, offset1Y,
                        offset2X, offset2Y,
                        region1X, region1Y,
                        region2X, region2Y,
                        channels.x, channels.y,
                        x, y, depth);
            }
                break;
            case DEPTH:
            	assert(false);
                featureResponse = calculateDepthFeature(0,
                        imageWidth, imageHeight,
                        offset1X, offset1Y,
                        offset2X, offset2Y,
                        region1X, region1Y,
                        region2X, region2Y,
                        x, y, depth);
                break;
        }

        float threshold = getThreshold(currentNodeOffset, tree);
        assert(!isnan(threshold));

		int value = static_cast<int>(!(featureResponse <= threshold));
        currentNodeOffset += leftNodeOffset + value;
    }

}

__global__ void normalizeProbabilitiesKernel(float* probabilities, int numLabels, int width, int height) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) {
        return;
    }
    const unsigned int y = blockIdx.y;
    assert(y < height);

    float sum = 0.0;
    for (int label = 0; label < numLabels; label++) {
        sum += probabilities[label * width * height + y * width + x];
    }

    if (sum == 0) {
        return;
    }

    for (int label = 0; label < numLabels; label++) {
        probabilities[label * width * height + y * width + x] /= sum;
    }

}

__global__ void maxProbabilitiesKernel(const float* probabilities, LabelType* output, int numLabels, int width,
        int height) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) {
        return;
    }
    const unsigned int y = blockIdx.y;
    assert(y < height);

    LabelType maxLabel = 0;
    float max = 0.0;
    for (LabelType label = 0; label < numLabels; label++) {
        const float probability = probabilities[label * width * height + y * width + x];
        if (probability > max) {
            max = probability;
            maxLabel = label;
        }
    }

    output[y * width + x] = maxLabel;
}

void normalizeProbabilities(cuv::ndarray<float, cuv::dev_memory_space>& probabilities) {

    utils::Profile profileClassifyImage("normalizeProbabilities");

    cudaStream_t stream = streams[0];

    const unsigned int numLabels = probabilities.shape(0);
    const unsigned int height = probabilities.shape(1);
    const unsigned int width = probabilities.shape(2);

    unsigned int threadsPerBlock = std::min(width, 128u);
    int blocks = std::ceil(width / static_cast<float>(threadsPerBlock));

    dim3 threads(threadsPerBlock);
    dim3 blockSize(blocks, height);

    normalizeProbabilitiesKernel<<<blockSize, threads, 0, stream>>>(probabilities.ptr(), numLabels, width, height);

    cudaSafeCall(cudaStreamSynchronize(stream));
}

void determineMaxProbabilities(const cuv::ndarray<float, cuv::dev_memory_space>& probabilities,
        cuv::ndarray<LabelType, cuv::dev_memory_space>& output) {

    utils::Profile profileClassifyImage("determineMaxProbabilities");

    const unsigned int numLabels = probabilities.shape(0);
    const unsigned int height = probabilities.shape(1);
    const unsigned int width = probabilities.shape(2);

    assert(output.shape(0) == height);
    assert(output.shape(1) == width);

    cudaStream_t stream = streams[0];

    unsigned int threadsPerBlock = std::min(width, 128u);
    int blocks = std::ceil(width / static_cast<float>(threadsPerBlock));

    dim3 threads(threadsPerBlock);
    dim3 blockSize(blocks, height);

    maxProbabilitiesKernel<<<blockSize, threads, 0, stream>>>(probabilities.ptr(), output.ptr(), numLabels, width, height);

    cudaSafeCall(cudaStreamSynchronize(stream));
}

void classifyImage(int treeCacheSize, cuv::ndarray<float, cuv::dev_memory_space>& output, const RGBDImage& image,
        LabelType numLabels, const boost::shared_ptr<const TreeNodes>& treeData, bool useDepthImages) {

    std::set<const RGBDImage*> images;
    images.insert(&image);

    tbb::mutex::scoped_lock lock(textureMutex);

    utils::Profile profileClassifyImage("classifyImage");

    imageCache.copyImages(1, images);

    cudaStream_t stream = streams[0];

    assert(output.shape(0) == numLabels);
    assert(output.shape(1) == static_cast<unsigned int>(image.getHeight()));
    assert(output.shape(2) == static_cast<unsigned int>(image.getWidth()));

    const int threadsPerRow = 8;
    const int threadsPerColumn = 16;
    int blocksX = std::ceil(image.getWidth() / static_cast<float>(threadsPerRow));
    int blocksY = std::ceil(image.getHeight() / static_cast<float>(threadsPerColumn));

    dim3 threads(threadsPerRow, threadsPerColumn);
    dim3 blockSize(blocksX, blocksY);

    treeCache.copyTree(treeCacheSize, treeData.get());

    size_t tree = treeCache.getElementPos(treeData.get());

    utils::Profile profileClassifyImageKernel("classifyImageKernel");

    cudaSafeCall(cudaFuncSetCacheConfig(classifyKernel, cudaFuncCachePreferL1));

    classifyKernel<<<blockSize, threads, 0, stream>>>(output.ptr(), tree,
            image.getWidth(), image.getHeight(),
            numLabels, useDepthImages);

    cudaSafeCall(cudaStreamSynchronize(stream));
}

__global__ void featureResponseKernel(
        FeatureResponseType* featureResponses1,
        FeatureResponseType* featureResponses2,
        const int8_t* types,
        const int16_t imageWidth, const int16_t imageHeight,
        const int8_t* offsets1X, const int8_t* offsets1Y,
        const int8_t* offsets2X, const int8_t* offsets2Y,
        const int8_t* regions1X, const int8_t* regions1Y,
        const int8_t* regions2X, const int8_t* regions2Y,
        const int8_t* channels1, const int8_t* channels2,
        const int* samplesX, const int* samplesY, const float* depths,
        const int* imageNumbers,  const bool* sampleFlipping, unsigned int numFeatures, unsigned int numSamples) {

    unsigned int feature = blockIdx.x * blockDim.y + threadIdx.y;
    unsigned int sample = blockIdx.y * blockDim.x + threadIdx.x;

    if (feature >= numFeatures || sample >= numSamples) {
        return;
    }

    int8_t type = types[feature];
    assert(type == COLOR || type == DEPTH);

    int8_t offset1X = offsets1X[feature];
    int8_t offset1Y = offsets1Y[feature];
    int8_t offset2X = offsets2X[feature];
    int8_t offset2Y = offsets2Y[feature];
    int8_t region1X = regions1X[feature];
    int8_t region1Y = regions1Y[feature];
    int8_t region2X = regions2X[feature];
    int8_t region2Y = regions2Y[feature];

    int imageNr = imageNumbers[sample];

    FeatureResponseType featureResponse1;
    FeatureResponseType featureResponse2 = 0;

    bool useFlipping = sampleFlipping[sample];

    switch (type) {
        case COLOR:
        { featureResponse1 = calculateColorFeature(imageNr,
                    imageWidth, imageHeight,
                    offset1X, offset1Y,
                    offset2X, offset2Y,
                    region1X, region1Y,
                    region2X, region2Y,
                    channels1[feature], channels2[feature],
                    samplesX[sample], samplesY[sample], depths[sample]);
          if (useFlipping) {
            featureResponse2 = calculateColorFeature(imageNr,
                     imageWidth, imageHeight,
                     -offset1X, offset1Y,
                     -offset2X, offset2Y,
                     region1X, region1Y,
                     region2X, region2Y,
                     channels1[feature], channels2[feature],
                     samplesX[sample], samplesY[sample], depths[sample]);}}
            break;
        case DEPTH:
        { featureResponse1 = calculateDepthFeature(imageNr,
                    imageWidth, imageHeight,
                    offset1X, offset1Y,
                    offset2X, offset2Y,
                    region1X, region1Y,
                    region2X, region2Y,
                    samplesX[sample], samplesY[sample], depths[sample]);
            if (useFlipping) {
            featureResponse2 = calculateDepthFeature(imageNr,
                    imageWidth, imageHeight,
                    -offset1X, offset1Y,
                    -offset2X, offset2Y,
                    region1X, region1Y,
                    region2X, region2Y,
                    samplesX[sample], samplesY[sample], depths[sample]);}}
            break;
        default:
            assert(false);
            break;
    }

    featureResponses1[featureResponseOffset(sample, feature, numSamples, numFeatures)] = featureResponse1;
    featureResponses2[featureResponseOffset(sample, feature, numSamples, numFeatures)] = featureResponse2;
}

// http://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
#ifndef NDEBUG
__device__
static bool isPowerOfTwo(size_t x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}
#endif

__global__ void scoreKernel(const WeightType* counters,
        const float* thresholds,
        unsigned int numThresholds,
        unsigned int numLabels,
        unsigned int numFeatures,
        const WeightType* allClasses,
        ScoreType* scores) {

    unsigned int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= numFeatures) {
        return;
    }

    unsigned int thresh = blockIdx.y;

    WeightType totals[2] = { 0, 0 };

    for (unsigned int label = 0; label < numLabels; label++) {
        for (unsigned int value = 0; value < 2; value++) {
            unsigned int cidx = counterOffset(label, value, thresh, feature, numLabels, numFeatures,
                    numThresholds);
            WeightType counter = counters[cidx];
            totals[value] += counter;
        }
    }

    const WeightType* leftClasses = counters
            + counterOffset(0, 0, thresh, feature, numLabels, numFeatures, numThresholds);

    const WeightType* rightClasses = counters
            + counterOffset(0, 1, thresh, feature, numLabels, numFeatures, numThresholds);

    assert(rightClasses == leftClasses + 1);

    unsigned int leftRightStride = 2;

#ifndef NDEBUG
    unsigned int off0 = counterOffset(0, 0, thresh, feature, numLabels, numFeatures, numThresholds);
    unsigned int off1 = counterOffset(1, 0, thresh, feature, numLabels, numFeatures, numThresholds);
    assert(leftRightStride == off1 - off0);
#endif

    ScoreType score = NormalizedInformationGainScore::calculateScore(numLabels, leftClasses, rightClasses,
            leftRightStride, allClasses, static_cast<ScoreType>(totals[0]), static_cast<ScoreType>(totals[1]));

    scores[thresh * numFeatures + feature] = score;
}

__global__ void aggregateHistogramsKernel(
        const FeatureResponseType* featureResponses1,
        const FeatureResponseType* featureResponses2,
        WeightType* counters,
        const float* thresholds,
        const uint8_t* sampleLabel,
        const bool* sampleFlipping,
        unsigned int numThresholds,
        unsigned int numLabels,
        unsigned int numFeatures,
        unsigned int numSamples
        ) {

#ifndef NDEBUG
    const unsigned int COUNTER_MAX = 0xFFFF;
#endif

    // shape: 2 * numLabels * threadsPerBlock
    extern __shared__ unsigned short counterShared[];

    unsigned int feature = blockIdx.y;
    unsigned int thresh = blockIdx.x;

    assert(feature < numFeatures);
    assert(thresh < numThresholds);

    unsigned int offset = thresh * numFeatures + feature;
    const float threshold = thresholds[offset];

    // initialize shared memory
    // every thread must initialize 2*numLabels counters with zero
    for (unsigned int i = threadIdx.x; i < 2 * numLabels * blockDim.x; i += blockDim.x) {
        counterShared[i] = 0;
    }

    __syncthreads();

    unsigned int labelFlags = 0;

    // iterate over all samples and increment the according counter in shared memory
    const FeatureResponseType* resultPtr1 = featureResponses1
            + featureResponseOffset(threadIdx.x, feature, numSamples, numFeatures);
    const FeatureResponseType* resultPtr2 = featureResponses2
            + featureResponseOffset(threadIdx.x, feature, numSamples, numFeatures);
    for (unsigned int sample = threadIdx.x; sample < numSamples; sample += blockDim.x) {

        FeatureResponseType featureResponse1 = *resultPtr1;
        resultPtr1 += blockDim.x; // need to change if featureResponseOffset calculation changes
        FeatureResponseType featureResponse2 = *resultPtr2;
        resultPtr2 += blockDim.x; // need to change if featureResponseOffset calculation changes

        uint8_t label = sampleLabel[sample];
        bool useFlipping = sampleFlipping[sample];
        assert(label < numLabels);

        assert(label < 32);
        labelFlags |= 1 << label;

        int value = static_cast<int>(!(featureResponse1 <= threshold));
        assert(value == 0 || value == 1);
        assert(counterShared[(2 * label) * blockDim.x + 2 * threadIdx.x + value] < COUNTER_MAX);
        counterShared[(2 * label) * blockDim.x + 2 * threadIdx.x + value]++;

        if (useFlipping){
        value =  static_cast<int>(!(featureResponse2 <= threshold));
        assert(value == 0 || value == 1);
        assert(counterShared[(2 * label) * blockDim.x + 2 * threadIdx.x + value] < COUNTER_MAX);
        counterShared[(2 * label) * blockDim.x + 2 * threadIdx.x + value]++;}
        // no need to sync here because data is accessed only by the same thread in this loop
    }

    // no sync needed here because it is done in the loop over the labels

    assert(isPowerOfTwo(blockDim.x));

    // reduce the 2*labels*threads counters in shared memory to 2*labels counters
    for (uint8_t label = 0; label < numLabels; label++) {

        // skip labels without samples
      /*  if (__syncthreads_or(labelFlags & (1 << label)) == 0) {
            if (threadIdx.x < 2) {
                counterShared[2 * label + threadIdx.x] = 0;
            }
            continue;
        }*/

        unsigned int idxA = (2 * label) * blockDim.x + threadIdx.x;
        for (unsigned int offset = blockDim.x; offset > 2; offset /= 2) {

            if (threadIdx.x < offset) {
                // check for counter overflow
                assert(COUNTER_MAX - counterShared[idxA] >= counterShared[idxA + offset]);

                counterShared[idxA] += counterShared[idxA + offset];
            }

            __syncthreads();
        }

        if (threadIdx.x < 2) {
            // write final result to a different (already unused) location in shared memory
            // this way, bank conflicts are avoided at the very end, when data is loaded from shared memory to write it to global memory in a coalesced manner
            counterShared[2 * label + threadIdx.x] = counterShared[idxA] + counterShared[idxA + 2];
        }

    }

    if (threadIdx.x < 2 * numLabels) {
        const unsigned int label = threadIdx.x / 2;
        const unsigned int value = threadIdx.x % 2;
        assert(threadIdx.x == 2 * label + value);
        const unsigned short count = counterShared[threadIdx.x];

        const unsigned int cidx = counterOffset(label, value, thresh, feature, numLabels, numFeatures,
                numThresholds);
        counters[cidx] += count;
    }
}

void ImageFeatureEvaluation::selectDevice() {
    int currentDeviceId;
    cudaSafeCall(cudaGetDevice(&currentDeviceId));

    const std::vector<int> deviceIds = configuration.getDeviceIds();
    const int targetDeviceId = deviceIds[treeId % deviceIds.size()];

    if (currentDeviceId != targetDeviceId) {
        CURFIL_DEBUG("tree " << treeId << ": switching from device " << currentDeviceId << " to " << targetDeviceId);
        cudaSafeCall(cudaSetDevice(targetDeviceId));
        cudaSafeCall(cudaGetDevice(&currentDeviceId));
        if (currentDeviceId != targetDeviceId) {
            throw std::runtime_error("failed to switch GPU device");
        }
    }
}

void ImageFeatureEvaluation::initDevice() {

    selectDevice();

    cudaDeviceProp prop;
    int currentDeviceId;
    cudaSafeCall(cudaGetDevice(&currentDeviceId));
    cudaSafeCall(cudaGetDeviceProperties(&prop, currentDeviceId));
    CURFIL_INFO("GPU Device " << currentDeviceId << ": " << prop.name);

    {
        tbb::mutex::scoped_lock initLock(initMutex);
        if (!initialized) {
            for (int i = 0; i < NUM_STREAMS; i++) {
                cudaSafeCall(cudaStreamCreate(&streams[i]));
            }
            initialized = true;
            CURFIL_DEBUG("created " << NUM_STREAMS << " streams");
        }
    }
}

static void addBatch(RandomTree<PixelInstance, ImageFeatureFunction>& node,
        std::vector<std::vector<const PixelInstance*> >& batches,
        std::vector<const PixelInstance*>& currentBatch,
        std::set<const RGBDImage*>& imagesInCurrentBatch) {
    assert(!currentBatch.empty());

    unsigned int batchNr = batches.size();

    std::set<LabelType> labels;
    for (size_t i = 0; i < currentBatch.size(); i++) {
        labels.insert(currentBatch[i]->getLabel());
    }

    node.setTimerAnnotation((boost::format("batch%d.numSamples") % batchNr).str(), currentBatch.size());
    node.setTimerAnnotation((boost::format("batch%d.numImages") % batchNr).str(), imagesInCurrentBatch.size());
    node.setTimerAnnotation((boost::format("batch%d.numLabels") % batchNr).str(), labels.size());

    CURFIL_DEBUG((boost::format("batch%d.numSamples: %d") % batchNr % currentBatch.size()).str());
    CURFIL_DEBUG((boost::format("batch%d.numImages: %d") % batchNr % imagesInCurrentBatch.size()).str());

    batches.push_back(currentBatch);

    currentBatch.clear();
    imagesInCurrentBatch.clear();
}

std::vector<std::vector<const PixelInstance*> > ImageFeatureEvaluation::prepare(
        const std::vector<const PixelInstance*>& hostSamples,
        RandomTree<PixelInstance, ImageFeatureFunction>& node, cuv::dev_memory_space, bool keepMutexLocked) {

    selectDevice();

    assert(hostSamples.size() > 0);

    textureMutex.lock();

    utils::Timer prepareTime;

    std::vector<const PixelInstance*> samples;

    // take samples with cached images first. then the uncached images
    for (size_t sample = 0; sample < hostSamples.size(); sample++) {
        if (imageCache.containsElement(hostSamples[sample]->getRGBDImage())) {
            samples.push_back(hostSamples[sample]);
        }
    }
    for (size_t sample = 0; sample < hostSamples.size(); sample++) {
        if (!imageCache.containsElement(hostSamples[sample]->getRGBDImage())) {
            samples.push_back(hostSamples[sample]);
        }
    }

    assert(samples.size() == hostSamples.size());

    std::vector<std::vector<const PixelInstance*> > batches;

    std::vector<const PixelInstance*> currentBatch;
    std::set<const RGBDImage*> imagesInCurrentBatch;
    for (size_t sampleNr = 0; sampleNr < samples.size(); sampleNr++) {

        const PixelInstance* sample = samples[sampleNr];

        assert(sample->getDepth().isValid());

        if ((imagesInCurrentBatch.find(sample->getRGBDImage()) == imagesInCurrentBatch.end()
                && imagesInCurrentBatch.size() >= static_cast<size_t>(configuration.getImageCacheSize()))
                || currentBatch.size() == configuration.getMaxSamplesPerBatch()) {
            addBatch(node, batches, currentBatch, imagesInCurrentBatch);
        }

        imagesInCurrentBatch.insert(sample->getRGBDImage());
        currentBatch.push_back(sample);
    }

    if (!currentBatch.empty()) {
        addBatch(node, batches, currentBatch, imagesInCurrentBatch);
    }

    assert(!batches.empty());

    imageWidth = batches[0][0]->width();
    imageHeight = batches[0][0]->height();

    node.setTimerValue("prepareBatches", prepareTime);

    if (!keepMutexLocked) {
        textureMutex.unlock();
    }

    return batches;
}

template<>
void ImageFeatureEvaluation::sortFeatures(
        ImageFeaturesAndThresholds<cuv::dev_memory_space>& featuresAndThresholds,
        const cuv::ndarray<int, cuv::dev_memory_space>& keysIndices) const {

    utils::Profile profile("sortFeatures");

    unsigned int numFeatures = configuration.getFeatureCount();

    ImageFeaturesAndThresholds<cuv::dev_memory_space> sortedFeaturesAndThresholds(numFeatures,
            configuration.getThresholds(), featuresAllocator);

    thrust::device_ptr<int> k(keysIndices[cuv::indices[0][cuv::index_range()]].ptr());
    thrust::device_ptr<int> i(keysIndices[cuv::indices[1][cuv::index_range()]].ptr());

    thrust::sort_by_key(k, k + numFeatures, i);

    cuv::ndarray<int8_t, cuv::dev_memory_space> features = featuresAndThresholds.features();
    cuv::ndarray<int8_t, cuv::dev_memory_space> sortedFeatures = sortedFeaturesAndThresholds.features();

    assert(features.shape() == sortedFeatures.shape());

    const size_t dim = features.shape(0);
    assert(dim == 11);
    for (size_t d = 0; d < dim; d++) {
        thrust::device_ptr<int8_t> ptr(features[cuv::indices[d][cuv::index_range()]].ptr());
        thrust::device_ptr<int8_t> sortedPtr(sortedFeatures[cuv::indices[d][cuv::index_range()]].ptr());
        thrust::gather(i, i + numFeatures, ptr, sortedPtr);
    }

    for (size_t thresh = 0; thresh < configuration.getThresholds(); thresh++) {
        thrust::device_ptr<float> thresholdsPtr(
                featuresAndThresholds.thresholds()[cuv::indices[thresh][cuv::index_range()]].ptr());
        thrust::device_ptr<float> sortedThresholdsPtr(
                sortedFeaturesAndThresholds.thresholds()[cuv::indices[thresh][cuv::index_range()]].ptr());

        thrust::gather(i, i + numFeatures, thresholdsPtr, sortedThresholdsPtr);

    }

    featuresAndThresholds = sortedFeaturesAndThresholds;
}

ImageFeaturesAndThresholds<cuv::dev_memory_space> ImageFeatureEvaluation::generateRandomFeatures(
        const std::vector<const PixelInstance*>& samples, int seed, const bool sort, cuv::dev_memory_space) {

    unsigned int numFeatures = configuration.getFeatureCount();
    unsigned int numThresholds = configuration.getThresholds();

    tbb::mutex::scoped_lock textureLock(textureMutex);

    Samples<cuv::dev_memory_space> samplesOnDevice = copySamplesToDevice(samples, streams[0]);

    ImageFeaturesAndThresholds<cuv::dev_memory_space> featuresAndThresholds(numFeatures, numThresholds,
            featuresAllocator);

    cuv::ndarray<int, cuv::dev_memory_space> keysIndices(2, numFeatures, keysIndicesAllocator);

    int threadsPerBlock = std::min(numFeatures, 128u);
    int blocks = std::ceil(numFeatures / static_cast<float>(threadsPerBlock));

    const size_t numSamples = samplesOnDevice.data.shape(1);
    assert(numSamples == samples.size());

    {

        cudaSafeCall(cudaFuncSetCacheConfig(generateRandomFeaturesKernel, cudaFuncCachePreferL1));

        utils::Profile profile("generateRandomFeatures");
        generateRandomFeaturesKernel<<<blocks, threadsPerBlock, 0, streams[0]>>>(seed,
                numFeatures,
                keysIndices[cuv::indices[0][cuv::index_range()]].ptr(),
                keysIndices[cuv::indices[1][cuv::index_range()]].ptr(),
                configuration.getBoxRadius(), configuration.getRegionSize(),
                featuresAndThresholds.types().ptr(),
                featuresAndThresholds.offset1X().ptr(), featuresAndThresholds.offset1Y().ptr(),
                featuresAndThresholds.region1X().ptr(), featuresAndThresholds.region1Y().ptr(),
                featuresAndThresholds.offset2X().ptr(), featuresAndThresholds.offset2Y().ptr(),
                featuresAndThresholds.region2X().ptr(), featuresAndThresholds.region2Y().ptr(),
                featuresAndThresholds.channel1().ptr(), featuresAndThresholds.channel2().ptr(),
                featuresAndThresholds.thresholds().ptr(),
                numThresholds,
                numSamples,
                imageWidth, imageHeight,
                samplesOnDevice.imageNumbers,
                samplesOnDevice.depths,
                samplesOnDevice.sampleX,
                samplesOnDevice.sampleY,
                samplesOnDevice.labels,
                configuration.isUseDepthImages()
        );
        if (profile.isEnabled()) {
            cudaSafeCall(cudaStreamSynchronize(streams[0]));
        }
    }

    if (sort) {
        sortFeatures(featuresAndThresholds, keysIndices);
    }

    cudaSafeCall(cudaStreamSynchronize(streams[0]));

    return featuresAndThresholds;
}

template<>
cuv::ndarray<WeightType, cuv::dev_memory_space> ImageFeatureEvaluation::calculateFeatureResponsesAndHistograms(
        RandomTree<PixelInstance, ImageFeatureFunction>& node,
        const std::vector<std::vector<const PixelInstance*> >& batches,
        const ImageFeaturesAndThresholds<cuv::dev_memory_space>& featuresAndThresholds,
        cuv::ndarray<FeatureResponseType, cuv::host_memory_space>* featureResponsesHost) {

    unsigned int numFeatures = configuration.getFeatureCount();
    unsigned int numThresholds = configuration.getThresholds();

    const size_t numLabels = node.getNumClasses();

#ifndef NDEBUG
    {
        size_t numLabelsCheck = 0;
        for (size_t batch = 0; batch < batches.size(); batch++) {
            for (size_t sample = 0; sample < batches[batch].size(); sample++) {
                numLabelsCheck = std::max(numLabelsCheck,
                        static_cast<size_t>(batches[batch][sample]->getLabel() + 1));
            }
        }
        if (numLabelsCheck > numLabels) {
            CURFIL_DEBUG("numLabelsCheck: " << numLabelsCheck);
            CURFIL_DEBUG("numLabels:      " << numLabels);
            assert(false);
        }
    }
#endif

    // see function counterOffset()
    // features × threshold × labels × 2
    std::vector<unsigned int> shape;
    shape.push_back(numFeatures);
    shape.push_back(numThresholds);
    shape.push_back(numLabels);
    shape.push_back(2);

    cuv::ndarray<WeightType, cuv::dev_memory_space> counters(shape, countersAllocator);
    cudaSafeCall(cudaMemsetAsync(counters.ptr(), 0,
            static_cast<size_t>(counters.size() * sizeof(WeightType)), streams[0]));

    assert(numFeatures == configuration.getFeatureCount());

    cuv::ndarray<FeatureResponseType, cuv::dev_memory_space> featureResponsesDevice1(numFeatures,
            configuration.getMaxSamplesPerBatch(), featureResponsesAllocator);
    cuv::ndarray<FeatureResponseType, cuv::dev_memory_space> featureResponsesDevice2(numFeatures,
            configuration.getMaxSamplesPerBatch(), featureResponsesAllocator);

    if (featureResponsesHost) {
        size_t totalSamples = 0;
        for (size_t batch = 0; batch < batches.size(); batch++) {
            totalSamples += batches[batch].size();
        }
        featureResponsesHost->resize(numFeatures, totalSamples);
    }

    size_t samplesProcessed = 0;
    {
        for (size_t batch = 0; batch < batches.size(); batch++) {
            const std::vector<const PixelInstance*>& currentBatch = batches[batch];
            unsigned int batchSize = currentBatch.size();

            if (batch > 0) {
                textureMutex.lock();
            }

            Samples<cuv::dev_memory_space> sampleData = copySamplesToDevice(currentBatch, streams[0]);

            featureResponsesDevice1.resize(numFeatures, batchSize);
            featureResponsesDevice2.resize(numFeatures, batchSize);

            unsigned int featuresPerBlock = std::min(numFeatures, 32u);
            unsigned int samplesPerBlock = std::min(batchSize, 4u);
            int featureBlocks = std::ceil(numFeatures / static_cast<float>(featuresPerBlock));
            int sampleBlocks = std::ceil(batchSize / static_cast<float>(samplesPerBlock));

            dim3 blockSize(featureBlocks, sampleBlocks);
            dim3 threads(samplesPerBlock, featuresPerBlock);

            CURFIL_DEBUG("feature response kernel: launching " << blockSize.x << "x" <<blockSize.y
                    << " blocks with " << threads.x << "x" << threads.y << " threads");

            cudaSafeCall(cudaStreamSynchronize(streams[0]));

            utils::Timer featureResponseTimer;

            {
                cudaSafeCall(cudaFuncSetCacheConfig(featureResponseKernel, cudaFuncCachePreferL1));
                utils::Profile profile("calculate feature responses");
                featureResponseKernel<<<blockSize, threads, 0, streams[0]>>>(
                        featureResponsesDevice1.ptr(),
                        featureResponsesDevice2.ptr(),
                        featuresAndThresholds.types().ptr(),
                        imageWidth, imageHeight,
                        featuresAndThresholds.offset1X().ptr(), featuresAndThresholds.offset1Y().ptr(),
                        featuresAndThresholds.offset2X().ptr(), featuresAndThresholds.offset2Y().ptr(),
                        featuresAndThresholds.region1X().ptr(), featuresAndThresholds.region1Y().ptr(),
                        featuresAndThresholds.region2X().ptr(), featuresAndThresholds.region2Y().ptr(),
                        featuresAndThresholds.channel1().ptr(), featuresAndThresholds.channel2().ptr(),
                        sampleData.sampleX, sampleData.sampleY, sampleData.depths, sampleData.imageNumbers, sampleData.useFlipping,
                        numFeatures,
                        batchSize
                );
                if (profile.isEnabled()) {
                    cudaSafeCall(cudaStreamSynchronize(streams[0]));
                }
            }

            cudaSafeCall(cudaStreamSynchronize(streams[0]));

            //please note that featureResponsesDevice2 was not added
            if (featureResponsesHost) {
                // append feature responses on device to the feature responses for our caller
                (*featureResponsesHost)[cuv::indices[cuv::index_range()][cuv::index_range(samplesProcessed,
                        samplesProcessed + batchSize)]] = featureResponsesDevice1;
            }

            node.addTimerValue("featureResponse", featureResponseTimer);
            node.setTimerValue((boost::format("batch%d.featureResponse") % batch).str(), featureResponseTimer);

            textureMutex.unlock();

            utils::Timer aggregateHistogramsTimer;

            assert(numLabels > 0);

            {
                int threadsPerBlock = 128;

                if (batchSize <= 3000) {
                    threadsPerBlock = 64;
                }

                dim3 blockSize(numThresholds, numFeatures);
                dim3 threads(threadsPerBlock);

                utils::Profile profile((boost::format("aggregate histograms (%d samples)") % batchSize).str());
                unsigned int sharedMemory = sizeof(unsigned short) * 2 * numLabels * threadsPerBlock;

                cudaSafeCall(cudaFuncSetCacheConfig(aggregateHistogramsKernel, cudaFuncCachePreferShared));

                aggregateHistogramsKernel<<<blockSize, threads, sharedMemory, streams[1]>>>(
                        featureResponsesDevice1.ptr(),
                        featureResponsesDevice2.ptr(),
                        counters.ptr(),
                        featuresAndThresholds.thresholds().ptr(),
                        sampleData.labels,
                        sampleData.useFlipping,
                        numThresholds,
                        numLabels,
                        numFeatures,
                        batchSize
                );

                if (profile.isEnabled()) {
                    cudaSafeCall(cudaStreamSynchronize(streams[1]));
                }
            }

            cudaSafeCall(cudaStreamSynchronize(streams[1]));
            node.addTimerValue("aggregateHistograms", aggregateHistogramsTimer);

            node.setTimerValue((boost::format("batch%d.aggregateHistograms") % batch).str(),
                    aggregateHistogramsTimer);

            samplesProcessed += batchSize;
        }
    }

    return counters;
}

template<>
cuv::ndarray<ScoreType, cuv::host_memory_space> ImageFeatureEvaluation::calculateScores(
        const cuv::ndarray<WeightType, cuv::dev_memory_space>& counters,
        const ImageFeaturesAndThresholds<cuv::dev_memory_space>& featuresAndThresholds,
        const cuv::ndarray<WeightType, cuv::dev_memory_space>& histogram) {

    const unsigned int numFeatures = configuration.getFeatureCount();
    const unsigned int numThresholds = configuration.getThresholds();

    cuv::ndarray<ScoreType, cuv::dev_memory_space> scores(numThresholds, numFeatures, scoresAllocator);

    const size_t numLabels = histogram.size();
    assert(counters.shape(2) == numLabels);
    assert(numLabels > 0);

    {
        int threadsPerBlock = std::min(numFeatures, 128u);
        int blocks = std::ceil(numFeatures / static_cast<float>(threadsPerBlock));
        dim3 threads(threadsPerBlock);
        dim3 blockSize(blocks, numThresholds);

        utils::Profile profile("score kernel");

        cudaSafeCall(cudaFuncSetCacheConfig(scoreKernel, cudaFuncCachePreferL1));

        scoreKernel<<<blockSize, threads, 0, streams[1]>>>(
                counters.ptr(),
                featuresAndThresholds.thresholds().ptr(),
                numThresholds,
                numLabels,
                numFeatures,
                histogram.ptr(),
                scores.ptr()
        );

        if (profile.isEnabled()) {
            cudaSafeCall(cudaStreamSynchronize(streams[1]));
        }

    }

    cuv::ndarray<ScoreType, cuv::host_memory_space> scoresCPU(scores, streams[1]);
    cudaSafeCall(cudaStreamSynchronize(streams[1]));

    return scoresCPU;
}

boost::shared_ptr<const TreeNodes> convertTree(
        const boost::shared_ptr<const RandomTreeImage>& randomTreeImage) {
    const boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> >& tree =
            randomTreeImage->getTree();

    utils::Profile profile("convertTree");

    TreeNodes treeNodes(tree);
    return boost::make_shared<const TreeNodes>(treeNodes);
}

}
