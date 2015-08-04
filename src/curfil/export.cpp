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
#include "export.h"

#include <boost/algorithm/string.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <set>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <vector>

#include "version.h"

namespace curfil {

RandomTreeExport::RandomTreeExport(const TrainingConfiguration& configuration, const std::string& outputFolder,
        const std::string& trainingFolder, bool verbose) :
        date(boost::posix_time::microsec_clock::local_time()), configuration(configuration),
                outputFolder(outputFolder), trainingFolder(trainingFolder), verbose(verbose) {

}

void RandomTreeExport::writeXY(boost::property_tree::ptree& pt, const std::string& name, const XY& xy) {
    // http://stackoverflow.com/questions/3751357/c-how-to-create-an-array-using-boostproperty-tree/3751677#3751677
    boost::property_tree::ptree array;

    boost::property_tree::ptree c1, c2;
    c1.put("", boost::lexical_cast<std::string>(xy.getX()));
    c2.put("", boost::lexical_cast<std::string>(xy.getY()));
    array.push_back(std::make_pair("", c1));
    array.push_back(std::make_pair("", c2));
    pt.put_child(name, array);
}

void RandomTreeExport::writeFeatureDetails(boost::property_tree::ptree& pt, const ImageFeatureFunction& feature) {

    pt.put("type", feature.getTypeString());

    switch (feature.getType()) {
        case FeatureType::DEPTH:
            break;
        case FeatureType::COLOR:
            pt.put<int>("channel1", feature.getChannel1());
            pt.put<int>("channel2", feature.getChannel2());
            break;
        default:
            assert(false);
            break;
    }

    writeXY(pt, "offset1", feature.getOffset1());
    writeXY(pt, "region1", feature.getRegion1());
    writeXY(pt, "offset2", feature.getOffset2());
    writeXY(pt, "region2", feature.getRegion2());
}

std::vector<std::map<std::string, std::string> > RandomTreeExport::parseProcCpuInfo() {
    std::ifstream infile("/proc/cpuinfo");
    std::string line;

    std::vector<std::map<std::string, std::string> > processors;
    int processor = 0;

    while (std::getline(infile, line))
    {
        size_t pos = line.find(":");
        if (pos == std::string::npos) {
            continue;
        }

        if (line.length() <= pos + 1) {
            continue;
        }

        std::string key = line.substr(0, pos - 1);
        std::string value = line.substr(pos + 1);

        boost::algorithm::trim(key);
        boost::algorithm::trim(value);

        if (value.empty()) {
            continue;
        }

        if (key == "processor") {
            std::istringstream(value) >> processor;
        } else {
            processors.resize(processor + 1);
            processors[processor][key] = value;
        }
    }
    return processors;
}

int RandomTreeExport::getPhysicalCores() {
    const std::vector<std::map<std::string, std::string> > processorInfos = parseProcCpuInfo();

    std::map<int, int> cpuCoresPerCPU;

    for (size_t i = 0; i < processorInfos.size(); i++) {
        int physicalId, cpuCores;
        std::istringstream(processorInfos[i].at("physical id")) >> physicalId;
        std::istringstream(processorInfos[i].at("cpu cores")) >> cpuCores;
        cpuCoresPerCPU[physicalId] = cpuCores;
    }

    int physicalCores = 0;
    std::map<int, int>::const_iterator it;
    for (it = cpuCoresPerCPU.begin(); it != cpuCoresPerCPU.end(); it++) {
        physicalCores += it->second;
    }

    return physicalCores;
}

int RandomTreeExport::getPhysicalCpus() {

    const std::vector<std::map<std::string, std::string> > processorInfos = parseProcCpuInfo();

    int maxPhysicalCpu = 0;

    for (size_t i = 0; i < processorInfos.size(); i++) {
        int physicalId;
        std::istringstream(processorInfos[i].at("physical id")) >> physicalId;
        maxPhysicalCpu = std::max(maxPhysicalCpu, physicalId + 1);
    }
    return maxPhysicalCpu;
}

boost::property_tree::ptree RandomTreeExport::getProcessorModelNames() {
    const std::vector<std::map<std::string, std::string> > processorInfos = parseProcCpuInfo();

    std::set<std::string> modelNames;
    for (const auto& processorInfo : processorInfos) {
        modelNames.insert(processorInfo.at("model name"));
    }

    boost::property_tree::ptree processorModels;
    for (const auto& modelName : modelNames) {
      boost::property_tree::ptree mn;
      mn.put("", modelName);
      processorModels.push_back(std::make_pair("", mn));
    }
    return processorModels;
}

void RandomTreeExport::writeTree(boost::property_tree::ptree& pt, const RandomTreeImage& tree) const {

    char hostname[1024];
    hostname[1023] = '\0';
    gethostname(hostname, 1023);

    pt.put("hostname", hostname);

#ifdef NDEBUG
    pt.put("buildType", "RELEASE");
#else
    pt.put("buildType", "DEBUG");
#endif

    pt.put("version", getVersion());
    pt.put("date", date);
    pt.put("randomSeed", configuration.getRandomSeed());
    pt.put("folderTraining", boost::filesystem::absolute(boost::filesystem::path(trainingFolder)).string());
    pt.put("numCores", get_nprocs());
    pt.put("numPhysicalCPUs", getPhysicalCpus());
    pt.put_child("processorModels", getProcessorModelNames());
    pt.put("accelerationMode", configuration.getAccelerationModeString());
    pt.put_child("deviceId", toPropertyTree(configuration.getDeviceIds()));
    pt.put("samplesPerImage", configuration.getSamplesPerImage());
    pt.put("featureCount", configuration.getFeatureCount());
    pt.put("thresholds", configuration.getThresholds());
    pt.put("boxRadius", configuration.getBoxRadius());
    pt.put("regionSize", configuration.getRegionSize());
    pt.put("maxDepth", configuration.getMaxDepth());
    pt.put("numThreads", configuration.getNumThreads());
    pt.put("minSampleCount", configuration.getMinSampleCount());
    pt.put("maxSamplesPerBatch", configuration.getMaxSamplesPerBatch());
    pt.put("maxImages", configuration.getMaxImages());
    pt.put("imageCacheSize", configuration.getImageCacheSize());
    pt.put("subsamplingType", configuration.getSubsamplingType());
    pt.put("useCIELab", configuration.isUseCIELab());
    pt.put("useDepthFilling", configuration.isUseDepthFilling());
    pt.put("useDepthImages", configuration.isUseDepthImages());
    pt.put_child("ignoredColors", toPropertyTree(configuration.getIgnoredColors()));

    const cuv::ndarray<WeightType, cuv::host_memory_space>& priorDistribution = tree.getClassLabelPriorDistribution();
    for (LabelType label = 0; label < priorDistribution.size(); label++) {
        const std::string key = boost::str(boost::format("classLabelPriorDistribution.%d") % static_cast<int>(label));
        pt.put(key, static_cast<size_t>(priorDistribution[label]));
    }

    boost::property_tree::ptree ptSub;
    writeTree(ptSub, *(tree.getTree()));
    pt.put_child("tree", ptSub);
}

void RandomTreeExport::writeTree(boost::property_tree::ptree& pt,
        const RandomTree<PixelInstance, ImageFeatureFunction>& tree) const {

    pt.put("id", tree.getNodeId());
    pt.put("level", tree.getLevel());
    pt.put("samples", tree.getNumTrainSamples());
    pt.put("leaf", tree.isLeaf());

    if (verbose) {
        std::map<std::string, double>::const_iterator it;
        for (const auto& it : tree.getTimerValues()) {
            pt.put(std::string("timers.") + it.first, it.second);
        }

        for (const auto& it : tree.getTimerAnnotations()) {
            pt.put(std::string("timers.") + it.first, it.second);
        }
    }

    for (size_t i = 0; i < tree.getHistogram().size(); i++) {
        auto color = LabelImage::decodeLabel(i);
        const std::string key = boost::str(
                boost::format("histogram.%s (%d)") % color.toString() % i);
        pt.put(key, static_cast<size_t>(tree.getHistogram()[i]));
    }

    std::map<const RGBDImage*, size_t> countsPerImage;
    for (const auto& sample : tree.getTrainSamples()) {
        countsPerImage[sample.getRGBDImage()]++;
    }

    if (verbose) {
        for (const auto& c : countsPerImage) {
            const std::string key = boost::str(boost::format("countsPerImage.%1%") % c.first);
            pt.put(key, c.second);
        }
    }

    if (!tree.isLeaf()) {
        auto split = tree.getSplit();
        pt.put("split.threshold", split.getThreshold());
        pt.put("split.score", split.getScore());
        pt.put("split.featureId", split.getFeatureId());

        boost::property_tree::ptree featureTree;
        writeFeatureDetails(featureTree, split.getFeature());
        pt.put_child("split.feature", featureTree);
    }

    if (tree.getLeft()) {
        boost::property_tree::ptree childTree;
        writeTree(childTree, *tree.getLeft());
        pt.put_child("left", childTree);
    }
    if (tree.getRight()) {
        boost::property_tree::ptree childTree;
        writeTree(childTree, *tree.getRight());
        pt.put_child("right", childTree);
    }
}

void RandomTreeExport::writeJSON(const RandomTreeImage& tree, size_t treeNr) const {

    boost::property_tree::ptree pt;

    writeTree(pt, tree);

    assert(!outputFolder.empty());
    const std::string filename = boost::str(boost::format("%s/tree%d.json.gz") % outputFolder % treeNr);

    boost::iostreams::filtering_ostream ostream;
    if (boost::algorithm::ends_with(filename, ".gz")) {
        ostream.push(boost::iostreams::gzip_compressor());
    }
    ostream.push(boost::iostreams::file_sink(filename));

    boost::property_tree::write_json(ostream, pt);

    ostream.strict_sync();

    double filesize = (boost::filesystem::file_size(filename)) / static_cast<double>(1024 * 1024);
    CURFIL_INFO("wrote " << filename << (boost::format(" (%.2f MB)") % filesize).str());
}
}
