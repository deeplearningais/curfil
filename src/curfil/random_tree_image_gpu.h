#ifndef CURFIL_RANDOM_TREE_IMAGE_GPU_H
#define CURFIL_RANDOM_TREE_IMAGE_GPU_H

#include <cuda_runtime_api.h>
#include <limits.h>
#include <map>
#include <set>
#include <vector_types.h>
#include <vector>

#include "image.h"
#include "random_tree_image.h"

namespace curfil {

static const int colorChannels = 3;
static const int depthChannels = 2;

static const int depthChannel = 0;
static const int depthValidChannel = 1;

static const unsigned int NODES_PER_TREE_LAYER = 2048;
static const unsigned int LAYERS_PER_TREE = 16;

class TreeNodes {

private:

    static const size_t offsetLeftNode = 0;
    static const size_t offsetTypes = 4;
    static const size_t offsetFeatures = offsetTypes + 4;
    static const size_t offsetChannels = offsetFeatures + 8;
    static const size_t offsetThreshold = offsetChannels + 4;
    static const size_t offsetHistograms = offsetThreshold + 4;

    size_t m_treeId;
    size_t m_numNodes;
    size_t m_numLabels;
    size_t m_sizePerNode;
    cuv::ndarray<int8_t, cuv::host_memory_space> m_data;

    template<class T>
    void setValue(size_t node, size_t offset, const T& value);

    void setLeftNodeOffset(size_t node, int offset);
    void setThreshold(size_t node, float threshold);
    void setHistogramValue(size_t node, size_t label, float value);
    void setType(size_t node, int8_t value);
    void setOffset1X(size_t node, int8_t value);
    void setOffset1Y(size_t node, int8_t value);
    void setRegion1X(size_t node, int8_t value);
    void setRegion1Y(size_t node, int8_t value);
    void setOffset2X(size_t node, int8_t value);
    void setOffset2Y(size_t node, int8_t value);
    void setRegion2X(size_t node, int8_t value);
    void setRegion2Y(size_t node, int8_t value);
    void setChannel1(size_t node, uint16_t value);
    void setChannel2(size_t node, uint16_t value);

    void convert(const boost::shared_ptr<const RandomTree<PixelInstance, ImageFeatureFunction> >& tree);

    TreeNodes& operator=(const TreeNodes& other);

public:

    TreeNodes(const TreeNodes& other);

    TreeNodes(const boost::shared_ptr<const RandomTree<PixelInstance, ImageFeatureFunction> >& tree);

    size_t getTreeId() const {
        return m_treeId;
    }

    size_t numNodes() const {
        return m_numNodes;
    }

    size_t numLabels() const {
        return m_numLabels;
    }

    size_t sizePerNode() const {
        return m_sizePerNode;
    }

    cuv::ndarray<int8_t, cuv::host_memory_space>& data() {
        return m_data;
    }

    const cuv::ndarray<int8_t, cuv::host_memory_space>& data() const {
        return m_data;
    }

};

class DeviceCache {

private:
    DeviceCache(const DeviceCache& other);
    DeviceCache& operator=(const DeviceCache& other);

public:

    virtual ~DeviceCache();

    std::map<const void*, size_t>& getIdMap() {
        return elementIdMap;
    }

    bool containsElement(const void* element) const;

    size_t getElementPos(const void* element) const;

    void clear();

    size_t getTotalTransferTimeMircoseconds() const {
        return totalTransferTimeMicroseconds;
    }

protected:

    DeviceCache() :
            cacheSize(0), elementIdMap(), elementTimes(), currentTime(0), bound(false), totalTransferTimeMicroseconds(0) {
    }

    bool isBound() const {
        return bound;
    }

    void setBound(bool bound);

    void updateCacheSize(size_t cacheSize);

    virtual void bind() = 0;
    virtual void unbind() = 0;

    virtual void allocArray() = 0;
    virtual void freeArray() = 0;

    size_t getCacheSize() const {
        return cacheSize;
    }

    void copyElements(size_t cacheSize, const std::set<const void*>& elements);

    virtual void transferElement(size_t pos, const void* element, cudaStream_t stream) = 0;

    // for logging
    virtual std::string getElementName(const void* element) const = 0;
    virtual std::string getElementsName() const = 0;

private:

    size_t cacheSize;
    std::map<const void*, size_t> elementIdMap;
    std::map<size_t, size_t> elementTimes;

    // poor man’s vector clock
    size_t currentTime;

    bool bound;

    size_t totalTransferTimeMicroseconds;

};

class ImageCache: public DeviceCache {

private:
    ImageCache(const ImageCache& other);
    ImageCache& operator=(const ImageCache& other);

public:

    ImageCache();

    virtual ~ImageCache();

    void copyImages(size_t imageCacheSize, const std::set<const RGBDImage*>& images);

    void copyImages(size_t imageCacheSize, const std::vector<const PixelInstance*>& samples);

protected:

    virtual void bind();
    virtual void unbind();

    virtual void allocArray();
    virtual void freeArray();

    virtual void transferElement(size_t pos, const void* element, cudaStream_t stream);
    virtual std::string getElementName(const void* element) const;
    virtual std::string getElementsName() const;

private:

    int width;
    int height;

    cudaArray* colorTextureData;
    cudaArray* depthTextureData;

};

class TreeCache: public DeviceCache {

private:
    TreeCache(const TreeCache& other);
    TreeCache& operator=(const TreeCache& other);

public:

    TreeCache();

    virtual ~TreeCache();

    void copyTree(size_t cacheSize, const TreeNodes* tree);

    void copyTrees(size_t cacheSize, const std::set<const TreeNodes*>& trees);

protected:

    virtual void transferElement(size_t elementPos, const void* element, cudaStream_t stream);
    virtual std::string getElementName(const void* element) const;
    virtual std::string getElementsName() const;

    virtual void bind();
    virtual void unbind();

    virtual void freeArray();
    virtual void allocArray();

private:

    size_t sizePerNode;
    LabelType numLabels;

    cudaArray* treeTextureData;
};

class RandomTreeImage;

class TreeNodeData {

public:
    int leftNodeOffset;
    int type;
    int8_t offset1X, offset1Y, region1X, region1Y;
    int8_t offset2X, offset2Y, region2X, region2Y;
    uint8_t channel1, channel2;
    float threshold;
    cuv::ndarray<float, cuv::host_memory_space> histogram;
};

// for the unit test
TreeNodeData getTreeNode(const int nodeNr, const boost::shared_ptr<const TreeNodes>& treeData);

boost::shared_ptr<const TreeNodes> convertTree(const boost::shared_ptr<const RandomTreeImage>& randomTreeImage);

void normalizeProbabilities(cuv::ndarray<float, cuv::dev_memory_space>& probabilities);

void determineMaxProbabilities(const cuv::ndarray<float, cuv::dev_memory_space>& probabilities,
        cuv::ndarray<LabelType, cuv::dev_memory_space>& output);

void classifyImage(int treeCacheSize, cuv::ndarray<float, cuv::dev_memory_space>& output, const RGBDImage& image,
        LabelType numLabels, const boost::shared_ptr<const TreeNodes>& treeData);

// for the unit test
void clearImageCache();

}

#endif
