#include <boost/program_options.hpp>
#include <iomanip>
#include <tbb/task_scheduler_init.h>

#include "export.hpp"
#include "image.h"
#include "random_tree_image_ensemble.h"
#include "random_tree_image.h"
#include "utils.h"
#include "version.h"

namespace po = boost::program_options;

using namespace curfil;

static RandomTreeImageEnsemble train(const std::string& folderTraining, size_t trees,
        const TrainingConfiguration& configuration, bool trainTreesSequentially) {

    INFO("training data from: " << folderTraining);
    INFO("trees: " << trees);
    INFO("training trees sequentially: " << trainTreesSequentially);
    INFO(configuration);

    std::vector<LabeledRGBDImage> trainLabelImages = loadImages(folderTraining, configuration.isUseCIELab(),
            configuration.isUseDepthFilling());
    auto filenames = listImageFilenames(folderTraining);
    if (filenames.empty()) {
        throw std::runtime_error(std::string("found no files in ") + folderTraining);
    }

    // Train

    RandomTreeImageEnsemble randomForest(trees, configuration);

    utils::Timer trainTimer;
    randomForest.train(trainLabelImages, trainTreesSequentially);
    trainTimer.stop();

    INFO("training took " << trainTimer.format(2) <<
            " (" << std::setprecision(3) << trainTimer.getSeconds() / 60.0 << " min)");

    std::cout << randomForest;
    for (const auto& featureCount : randomForest.countFeatures()) {
        const std::string featureType = featureCount.first;
        INFO("feature " << featureType << ": " << featureCount.second);
    }

    return randomForest;
}

int main(int argc, char **argv) {

    std::string folderTraining;
    std::string outputFolder;
    int trees;
    unsigned int samplesPerImage;
    unsigned int featureCount;
    unsigned int minSampleCount;
    int maxDepth;
    uint16_t boxRadius;
    uint16_t regionSize;
    uint16_t numThresholds;
    std::string modeString;
    int numThreads;
    std::string subsamplingType;
    bool profiling;
    bool useCIELab;
    bool useDepthFilling;
    std::vector<int> deviceIds;
    int maxImages;
    int imageCacheSize;
    unsigned int maxSamplesPerBatch;
    int randomSeed;
    std::vector<std::string> ignoredColors;
    bool trainTreesSequentially;
    bool verboseTree;

    // Declare the supported options.
    po::options_description options("options");
    options.add_options()
    ("help", "produce help message")
    ("version", "show version and exit")
    ("folderTraining", po::value<std::string>(&folderTraining)->required(), "folder with training images")
    ("trees", po::value<int>(&trees)->required(), "number of trees to train")
    ("samplesPerImage", po::value<unsigned int>(&samplesPerImage)->required(), "samples per image")
    ("featureCount", po::value<unsigned int>(&featureCount)->required(), "feature count")
    ("minSampleCount", po::value<unsigned int>(&minSampleCount)->required(), "min samples count")
    ("maxDepth", po::value<int>(&maxDepth)->required(), "maximum tree depth")
    ("boxRadius", po::value<uint16_t>(&boxRadius)->required(), "box radius")
    ("regionSize", po::value<uint16_t>(&regionSize)->required(), "region size")
    ("numThresholds", po::value<uint16_t>(&numThresholds)->required(), "number of thresholds to evaluate")
    ("outputFolder", po::value<std::string>(&outputFolder)->default_value(""), "folder to output predictions and trees")
    ("numThreads", po::value<int>(&numThreads)->default_value(tbb::task_scheduler_init::default_num_threads()),
            "number of threads")
    ("useCIELab", po::value<bool>(&useCIELab)->implicit_value(true)->default_value(true),
            "convert images to CIElab color space")
    ("useDepthFilling", po::value<bool>(&useDepthFilling)->implicit_value(true)->default_value(false),
            "whether to do simple depth filling")
    ("deviceId", po::value<std::vector<int> >(&deviceIds), "GPU device id (multiple occurence possible)")
    ("subsamplingType", po::value<std::string>(&subsamplingType)->default_value("classUniform"),
            "subsampling type: 'pixelUniform' or 'classUniform'")
    ("maxImages", po::value<int>(&maxImages)->default_value(0),
            "maximum number of images to load for training. set to 0 if all images should be loaded")
    ("imageCacheSize", po::value<int>(&imageCacheSize)->default_value(100), "number of images to keep on device")
    ("maxSamplesPerBatch", po::value<unsigned int>(&maxSamplesPerBatch)->default_value(5000u),
            "max number of samples per batch")
    ("mode", po::value<std::string>(&modeString)->default_value("gpu"), "mode: 'gpu' (default), 'cpu' or 'compare'")
    ("profile", po::value<bool>(&profiling)->implicit_value(true)->default_value(false), "profiling")
    ("randomSeed", po::value<int>(&randomSeed)->default_value(4711), "random seed")
    ("ignoreColor", po::value<std::vector<std::string> >(&ignoredColors),
            "do not sample pixels of this color. format: R,G,B where 0 <= R,G,B <= 255")
    ("verboseTree", po::value<bool>(&verboseTree)->implicit_value(true)->default_value(false),
            "whether to write verbose tree include profiling and debugging information")
    ("trainTreesSequentially", po::value<bool>(&trainTreesSequentially)->implicit_value(true)->default_value(false),
            "whether to train trees sequentially");
    ;

    po::positional_options_description pod;
    pod.add("folderTraining", 1);
    pod.add("trees", 1);
    pod.add("samplesPerImage", 1);
    pod.add("featureCount", 1);
    pod.add("minSampleCount", 1);
    pod.add("maxDepth", 1);
    pod.add("boxRadius", 1);
    pod.add("regionSize", 1);
    pod.add("numThresholds", 1);
    pod.add("outputFolder", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).positional(pod).options(options).run(), vm);

    if (argc <= 1 || vm.count("help")) {
        std::cout << options << std::endl;
        return EXIT_FAILURE;
    }

    if (argc <= 1 || vm.count("version")) {
        std::cout << argv[0] << " version " << getVersion() << std::endl;
        return EXIT_FAILURE;
    }

    try {
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    logVersionInfo();

    if (deviceIds.empty()) {
        INFO("no GPU device ID specified. using device 0.");
        deviceIds.push_back(0);
    }

    INFO("acceleration mode: " << modeString);
    INFO("CIELab: " << useCIELab);
    INFO("DepthFilling: " << useDepthFilling);

    utils::Profile::setEnabled(profiling);

    tbb::task_scheduler_init init(numThreads);

    TrainingConfiguration configuration(randomSeed, samplesPerImage, featureCount, minSampleCount,
            maxDepth, boxRadius, regionSize, numThresholds, numThreads, maxImages, imageCacheSize, maxSamplesPerBatch,
            TrainingConfiguration::parseAccelerationModeString(modeString), useCIELab, useDepthFilling, deviceIds,
            subsamplingType, ignoredColors);

    RandomTreeImageEnsemble forest = train(folderTraining,
            trees, configuration, trainTreesSequentially);

    if (!outputFolder.empty()) {
        RandomTreeExport treeExport(configuration, outputFolder, folderTraining, verboseTree);
        treeExport.writeJSON(forest);
    } else {
        WARNING("no output folder given. skipping JSON export");
    }

    INFO("finished");
    return EXIT_SUCCESS;
}

