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
    int maxImages;
    int imageCacheSize;
    unsigned int maxSamplesPerBatch;
    int randomSeed;
    int numThreads;
    std::string subsamplingType;
    std::vector<std::string> ignoredColors;
    bool useCIELab;
    bool useDepthFilling;
    std::vector<int> deviceIds;
    bool profiling;
    std::string lossFunction;

    // Declare the supported options.
    po::options_description options("options");
    options.add_options()
    ("help", "produce help message")
    ("version", "show version and exit")
    ("url", po::value<std::string>(&url)->required(), "MongoDB url")
    ("db", po::value<std::string>(&db)->required(), "database name")
    ("deviceId", po::value<std::vector<int> >(&deviceIds), "GPU device id (multiple occurrences possible)")
    ("experiment", po::value<std::string>(&experiment)->required(), "experiment name")
    ("trainingFolder", po::value<std::string>(&trainingFolder)->required(), "folder with training images")
    ("testingFolder", po::value<std::string>(&testingFolder)->required(), "folder with testing images")
    ("maxImages", po::value<int>(&maxImages)->default_value(0),
            "maximum number of images to load for training. set to 0 if all images should be loaded")
    ("imageCacheSize", po::value<int>(&imageCacheSize)->default_value(100), "number of images to keep on device")
    ("subsamplingType", po::value<std::string>(&subsamplingType), "subsampling type: 'pixelUniform' or 'classUniform'")
    ("maxSamplesPerBatch", po::value<unsigned int>(&maxSamplesPerBatch)->default_value(5000u),
            "max number of samples per batch")
    ("useCIELab", po::value<bool>(&useCIELab)->implicit_value(true)->required(), "convert images to CIE lab space")
    ("useDepthFilling", po::value<bool>(&useDepthFilling)->implicit_value(true)->default_value(false),
            "whether to do simple depth filling")
    ("randomSeed", po::value<int>(&randomSeed)->default_value(4711), "random seed")
    ("profile", po::value<bool>(&profiling)->implicit_value(true)->default_value(false), "profiling")
    ("ignoredColors", po::value<std::vector<std::string> >(&ignoredColors), "do not sample pixels of this color. Format: R,G,B in the range 0-255.")

    ("lossFunction", po::value<std::string>(&lossFunction),
            "measure the loss function should be based on. one of 'classAccuracy', 'classAccuracyNoVoid', 'pixelAccuracy', 'pixelAccuracyNoBackground', 'pixelAccuracyNoVoid'")
    ("numThreads", po::value<int>(&numThreads)->default_value(tbb::task_scheduler_init::default_num_threads()),
            "number of threads")
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

    INFO("used loss function is " << lossFunction);

    tbb::task_scheduler_init init(numThreads);

    const auto trainImages = loadImages(trainingFolder, useCIELab, useDepthFilling);
    const auto testImages = loadImages(testingFolder, useCIELab, useDepthFilling);

    HyperoptClient client(trainImages, testImages, useCIELab, useDepthFilling, deviceIds, maxImages, imageCacheSize,
            maxSamplesPerBatch, randomSeed, numThreads, subsamplingType, ignoredColors, lossFunction, url, db,
            BSON("exp_key" << experiment));
    client.run();

    INFO("finished");
    return EXIT_SUCCESS;
}

