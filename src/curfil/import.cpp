#include "import.h"

#include <boost/algorithm/string.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace curfil {

XY RandomTreeImport::readXY(const boost::property_tree::ptree& pt) {
    if (pt.size() != 2) {
        throw std::runtime_error(boost::str(boost::format("unexpected size: %d") % pt.size()));
    }

    int x = pt.front().second.get_value<int>();
    int y = pt.back().second.get_value<int>();
    return XY(x, y);
}

SplitFunction<PixelInstance, ImageFeatureFunction> RandomTreeImport::parseSplit(const boost::property_tree::ptree& pt) {

    float threshold = pt.get<float>("threshold");
    ScoreType score = pt.get<ScoreType>("score");
    size_t featureId = pt.get<size_t>("featureId");

    const boost::property_tree::ptree featureTree = pt.get_child("feature");

    const std::string featureTypeName = featureTree.get<std::string>("type");
    FeatureType featureType;
    if (featureTypeName == "color") {
        featureType = FeatureType::COLOR;
    }
    else if (featureTypeName == "depth") {
        featureType = FeatureType::DEPTH;
    } else {
        throw std::runtime_error(std::string("illegal feature type: " + featureTypeName));
    }

    Offset offset1, offset2;
    Region region1, region2;

    if (featureTypeName == "color" || featureTypeName == "depth") {
        offset1 = readXY(featureTree.get_child("offset1"));
        offset2 = readXY(featureTree.get_child("offset2"));

        region1 = readXY(featureTree.get_child("region1"));
        region2 = readXY(featureTree.get_child("region2"));
    }

    uint8_t channel1 = 0;
    uint8_t channel2 = 0;

    if (featureType == FeatureType::COLOR) {
        channel1 = featureTree.get<uint8_t>("channel1");
        channel2 = featureTree.get<uint8_t>("channel2");
    }

    ImageFeatureFunction feature(featureType, offset1, region1, channel1, offset2, region2, channel2);

    SplitFunction<PixelInstance, ImageFeatureFunction> split(featureId, feature, threshold, score);
    return split;
}

boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> > RandomTreeImport::readTree(
        const boost::property_tree::ptree& pt,
        const boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> >& parent) {
    int id = pt.get<int>("id");
    int level = pt.get<int>("level");
    WeightType numSamples = pt.get<WeightType>("samples");

    WeightType sumHistogram = 0;
    const boost::property_tree::ptree histogramChild = pt.get_child("histogram");
    const size_t numClasses = histogramChild.size();

    std::vector<WeightType> histogram(numClasses, 0);

    boost::property_tree::ptree::const_iterator it;
    for (it = histogramChild.begin(); it != histogramChild.end(); it++) {

        // key is in the format: "255,255,255 (1)"
        const std::string& key = it->first;

        RGBColor color;
        try {
            color = RGBColor(key.substr(0, key.find(" ")));
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("failed to parse histogram key: '") + key + "': " + e.what());
        }

        size_t start = key.find(" (") + 2;
        size_t stop = key.find(")", start);
        const std::string labelString = key.substr(start, stop - start);

        const LabelType label = getOrAddColorId(color, boost::lexical_cast<int>(labelString));
        assert(label < numClasses);

        if (histogram[label] != 0) {
            std::ostringstream o;
            o << "node: " << id << ": ";
            o << "illegal histogram state for " << key <<" (label: " << label << "): " << histogram[label];
            throw std::runtime_error(o.str());
        }

        const WeightType num = histogramChild.get<WeightType>(key);
        histogram[label] = num;

        sumHistogram += num;
    }

   // TODO: change it back to lte
    if (sumHistogram < numSamples) {
        throw std::runtime_error("incorrect histogram sum");
    }

    boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> > tree = boost::make_shared<
            RandomTree<PixelInstance, ImageFeatureFunction> >(id, level, parent, histogram);

    if (pt.find("left") != pt.not_found()) {
        const auto split = parseSplit(pt.get_child("split"));
        auto left = readTree(pt.get_child("left"), tree);
        auto right = readTree(pt.get_child("right"), tree);
        tree->addChildren(split, left, right);
    }

    return tree;
}

cuv::ndarray<WeightType, cuv::host_memory_space> RandomTreeImport::readClassLabelPriorDistribution(
        const boost::property_tree::ptree& p) {
    cuv::ndarray<WeightType, cuv::host_memory_space> classLabelPriorDistribution(p.size());

    boost::property_tree::ptree::const_iterator it;
    for (it = p.begin(); it != p.end(); it++) {
        const std::string& key = it->first;
        classLabelPriorDistribution[boost::lexical_cast<int>(key)] = p.get<size_t>(key);
    }

    return classLabelPriorDistribution;
}

TrainingConfiguration RandomTreeImport::readJSON(const std::string& filename,
        boost::shared_ptr<RandomTreeImage>& tree,
        std::string& hostname,
        boost::filesystem::path& folderTraining,
        boost::posix_time::ptime& date) {

    if (!boost::filesystem::is_regular_file(filename)) {
        throw std::runtime_error(std::string("failed to read tree file: '") + filename + "' is not a regular file");
    }

    boost::property_tree::ptree pt;

    boost::iostreams::filtering_istream istream;

    if (boost::algorithm::ends_with(filename, ".gz")) {
        istream.push(boost::iostreams::gzip_decompressor());
    }
    istream.push(boost::iostreams::file_source(filename));

    boost::property_tree::read_json(istream, pt);

    date = boost::posix_time::time_from_string(pt.get<std::string>("date"));
    folderTraining = pt.get<boost::filesystem::path>("folderTraining");
    hostname = pt.get<std::string>("hostname");
    int randomSeed = pt.get<int>("randomSeed");
    unsigned int samplesPerImage = pt.get<int>("samplesPerImage");
    unsigned int featureCount = pt.get<int>("featureCount");
    unsigned int minSampleCount = pt.get<int>("minSampleCount");
    int maxDepth = pt.get<int>("maxDepth");
    uint16_t boxRadius = pt.get<uint16_t>("boxRadius");
    uint16_t regionSize = pt.get<uint16_t>("regionSize");
    uint16_t thresholds = pt.get<uint16_t>("thresholds");
    int numThreads = pt.get<int>("numThreads");
    int imageCacheSize = pt.get<int>("imageCacheSize");

    bool useCIELab = true;
    const boost::optional<bool> useCIELabValue = pt.get_optional<bool>("useCIELab");
    if (useCIELabValue) {
        useCIELab = useCIELabValue.get();
    }

    bool useDepthFilling = false;
    const boost::optional<bool> useDepthFillingValue = pt.get_optional<bool>("useDepthFilling");
    if (useDepthFillingValue) {
        useDepthFilling = useDepthFillingValue.get();
    }

    int maxImages = 0;
    const boost::optional<int> maxImagesValue = pt.get_optional<int>("maxImages");
    if (maxImagesValue) {
        maxImages = maxImagesValue.get();
    }

    const boost::optional<std::string> subsamplingTypeValue = pt.get_optional<std::string>("subsamplingType");
    std::string subsamplingType = "pixelUniform";
    if (subsamplingTypeValue) {
        subsamplingType = subsamplingTypeValue.get();
    }

    unsigned int maxSamplesPerBatch = pt.get<unsigned int>("maxSamplesPerBatch");
    const std::string accelerationModeString = pt.get<std::string>("accelerationMode");

    const cuv::ndarray<WeightType, cuv::host_memory_space> classLabelPriorDistribution =
            readClassLabelPriorDistribution(
                    pt.get_child("classLabelPriorDistribution"));

    const std::vector<std::string> ignoredColors = fromPropertyTree<std::string>(
            pt.get_child_optional("ignoredColors"));

    const std::vector<int> deviceIds = fromPropertyTree(pt.get_child_optional("deviceIds"), std::vector<int>(1, 0));

    bool useDepthImages = true;
    const boost::optional<bool> useDepthImagesValue = pt.get_optional<bool>("useDepthImages");
    if (useDepthImagesValue) {
    	useDepthImages = useDepthImagesValue.get();
    }

    //TODO: remove
    //useDepthImages = false;

    TrainingConfiguration configuration(randomSeed, samplesPerImage, featureCount, minSampleCount, maxDepth, boxRadius,
            regionSize, thresholds, numThreads, maxImages, imageCacheSize, maxSamplesPerBatch,
            TrainingConfiguration::parseAccelerationModeString(accelerationModeString), useCIELab, useDepthFilling,
            deviceIds, subsamplingType, ignoredColors, useDepthImages);

    boost::shared_ptr<RandomTree<PixelInstance, ImageFeatureFunction> > randomTree = readTree(pt.get_child("tree"));
    assert(randomTree->isRoot());

    tree = boost::make_shared<RandomTreeImage>(randomTree, configuration, classLabelPriorDistribution);

    return configuration;
}

}
