#include "predict.h"

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <iomanip>
#include <tbb/task_scheduler_init.h>

#include "import.h"
#include "random_forest_image.h"
#include "random_tree_image.h"
#include "utils.h"
#include "version.h"

namespace po = boost::program_options;

using namespace curfil;

static void initDevice(int deviceId) {
    cudaDeviceProp prop;

    int currentDeviceId;
    cudaGetDevice(&currentDeviceId);
    if (deviceId != currentDeviceId) {
        CURFIL_INFO("switching from device " << currentDeviceId << " to " << deviceId)
        cudaSafeCall(cudaSetDevice(deviceId));
    }
    cudaSafeCall(cudaGetDeviceProperties(&prop, deviceId));
    CURFIL_INFO("GPU Device: " << prop.name);
}

int main(int argc, char **argv) {

    std::string folderPrediction = "";
    std::string folderTesting;
    std::vector<std::string> treeFiles;
    int numThreads;
    double histogramBias = 0.0;
    bool profiling = false;
    std::string modeString = "gpu";
    int deviceId = 0;
    bool useDepthFillingOption = false;
    bool writeProbabilityImages = false;

    // Declare the supported options.
    po::options_description options("options");
    options.add_options()
    ("help", "produce help message")
    ("version", "show version and exit")
    ("folderPrediction", po::value<std::string>(&folderPrediction)->default_value(folderPrediction),
            "folder to output prediction images. leave it empty to suppress writing of prediction images")
    ("folderTesting", po::value<std::string>(&folderTesting)->required(), "folder with test images")
    ("treeFile", po::value<std::vector<std::string> >(&treeFiles)->required(), "serialized tree(s) (JSON)")
    ("histogramBias", po::value<double>(&histogramBias)->default_value(histogramBias), "histogram bias")
    ("numThreads", po::value<int>(&numThreads)->default_value(tbb::task_scheduler_init::default_num_threads()),
            "number of threads")
    ("mode", po::value<std::string>(&modeString)->default_value(modeString), "mode: 'cpu' or 'gpu'")
    ("deviceId", po::value<int>(&deviceId)->default_value(deviceId), "GPU device id")
    ("profile", po::value<bool>(&profiling)->implicit_value(true)->default_value(profiling), "profiling")
    ("useDepthFilling",
            po::value<bool>(&useDepthFillingOption)->implicit_value(true)->default_value(useDepthFillingOption),
            "whether to do simple depth filling")
    ("writeProbabilityImages",
            po::value<bool>(&writeProbabilityImages)->implicit_value(true)->default_value(writeProbabilityImages),
            "whether to write probability PNGs of the prediction")
            ;

    po::positional_options_description pod;
    pod.add("folderPrediction", 1);
    pod.add("folderTesting", 1);
    pod.add("treeFile", -1);

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

    if (histogramBias < 0.0 || histogramBias >= 1.0) {
        throw std::runtime_error(boost::str(boost::format("illegal histogram bias: %lf") % histogramBias));
    }

    if (writeProbabilityImages && folderPrediction.empty()) {
        throw std::runtime_error("specified to write probability images but prediction folder was not set");
    }

    CURFIL_INFO("histogramBias: " << histogramBias);
    CURFIL_INFO("writeProbabilityImages: " << writeProbabilityImages);

    utils::Profile::setEnabled(profiling);

    tbb::task_scheduler_init init(numThreads);

    std::vector<int> deviceIds(1, deviceId);

    initDevice(deviceId);

    AccelerationMode mode = TrainingConfiguration::parseAccelerationModeString(modeString);
    RandomForestImage randomForest(treeFiles, deviceIds, mode, histogramBias);

    bool useDepthFilling = randomForest.getConfiguration().isUseDepthFilling();
    if (vm.count("useDepthFilling")) {
        if (useDepthFillingOption != useDepthFilling) {
            if (useDepthFilling) {
                CURFIL_WARNING("random forest was trained with depth filling. but prediction is performed without!");
            } else {
                CURFIL_WARNING(
                        "random forest was NOT trained with depth filling. but prediction is performed with depth filling!");
            }
        }
        // override
        useDepthFilling = useDepthFillingOption;
    }

    test(randomForest, folderTesting, folderPrediction, useDepthFilling, writeProbabilityImages);

    CURFIL_INFO("finished");
    return EXIT_SUCCESS;
}

