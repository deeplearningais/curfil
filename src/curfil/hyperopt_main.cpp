#include <boost/program_options.hpp>
#include <tbb/task_scheduler_init.h>

#include "hyperopt.h"
#include "image.h"
#include "utils.h"
#include "version.h"

namespace po = boost::program_options;

using namespace curfil;

int main(int argc, char **argv) {

    std::string url;
    std::string db;
    std::string experiment;
    std::string trainingFolder;
    std::string testingFolder;
    int maxImages = 0;
    int imageCacheSizeMB = 0;
    int randomSeed = 4711;
    int numThreads;
    std::string subsamplingType;
    std::vector<std::string> ignoredColors;
    bool useCIELab = true;
    bool useDepthFilling = false;
    int deviceId = 0;
    bool profiling = false;
    bool useDepthImages = true;
    std::string lossFunction;

    // Declare the supported options.
    po::options_description options("options");
    options.add_options()
    ("help", "produce help message")
    ("version", "show version and exit")
    ("url", po::value<std::string>(&url)->required(), "MongoDB url")
    ("db", po::value<std::string>(&db)->required(), "database name")
    ("deviceId", po::value<int>(&deviceId)->default_value(deviceId), "GPU device id")
    ("experiment", po::value<std::string>(&experiment)->required(), "experiment name")
    ("trainingFolder", po::value<std::string>(&trainingFolder)->required(), "folder with training images")
    ("testingFolder", po::value<std::string>(&testingFolder)->required(), "folder with testing images")
    ("maxImages", po::value<int>(&maxImages)->default_value(maxImages),
            "maximum number of images to load for training. set to 0 if all images should be loaded")
    ("imageCacheSize", po::value<int>(&imageCacheSizeMB)->default_value(imageCacheSizeMB),
            "image cache size on GPU in MB. 0 means automatic adjustment")
    ("subsamplingType", po::value<std::string>(&subsamplingType), "subsampling type: 'pixelUniform' or 'classUniform'")
    ("useCIELab", po::value<bool>(&useCIELab)->default_value(useCIELab), "convert images to CIE lab space")
    ("useDepthFilling", po::value<bool>(&useDepthFilling)->implicit_value(true)->default_value(useDepthFilling),
            "whether to do simple depth filling")
    ("randomSeed", po::value<int>(&randomSeed)->default_value(randomSeed), "random seed")
    ("profile", po::value<bool>(&profiling)->implicit_value(true)->default_value(profiling), "profiling")
    ("ignoreColor", po::value<std::vector<std::string> >(&ignoredColors),
            "do not sample pixels of this color. Format: R,G,B in the range 0-255.")

    ("lossFunction", po::value<std::string>(&lossFunction)->required(),
            "measure the loss function should be based on. one of 'classAccuracy', 'classAccuracyWithoutVoid', 'pixelAccuracy', 'pixelAccuracyWithoutVoid'")
    ("numThreads", po::value<int>(&numThreads)->default_value(tbb::task_scheduler_init::default_num_threads()),
            "number of threads")
    ("useDepthImages", po::value<bool>(&useDepthImages)->implicit_value(true)->default_value(useDepthImages),
                    "whether to use depth images")
            ;

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(options).run(), vm);

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

    utils::Profile::setEnabled(profiling);

    CURFIL_INFO("used loss function is " << lossFunction);

    tbb::task_scheduler_init init(numThreads);

    size_t numLabels;
    size_t numLabelsTesting;

    const auto trainImages = loadImages(trainingFolder, useCIELab, useDepthImages, useDepthFilling, ignoredColors, numLabels);
    const auto testImages = loadImages(testingFolder, useCIELab, useDepthImages, useDepthFilling, ignoredColors, numLabelsTesting);

    // currently training on only on GPU is tested
    std::vector<int> deviceIds(1, deviceId);

    HyperoptClient client(trainImages, testImages, useCIELab, useDepthFilling, deviceIds, maxImages, imageCacheSizeMB,
            randomSeed, numThreads, subsamplingType, ignoredColors, useDepthImages, numLabels, lossFunction, url, db,
            BSON("exp_key" << experiment));
    client.run();

    CURFIL_INFO("finished");
    return EXIT_SUCCESS;
}

