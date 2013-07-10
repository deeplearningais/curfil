#include "predict.h"

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <iomanip>
#include <tbb/task_scheduler_init.h>

#include "import.hpp"
#include "random_tree_image_ensemble.h"
#include "random_tree_image.h"
#include "utils.h"
#include "version.h"

namespace po = boost::program_options;

using namespace curfil;

static void initDevice() {
    cudaDeviceProp prop;

    int deviceId;
    cudaGetDevice(&deviceId);
    if (deviceId != 0) {
        INFO("switching from device " << deviceId << " to " << 0)
        deviceId = 0;
        cudaSafeCall(cudaSetDevice(deviceId));
    }
    cudaSafeCall(cudaGetDeviceProperties(&prop, deviceId));
    INFO("GPU Device: " << prop.name);
}

int main(int argc, char **argv) {

    std::string folderPrediction;
    std::string folderTesting;
    std::vector<std::string> treeFiles;
    int maxDepth;
    int numThreads;
    double histogramBias;
    bool profiling;
    std::string modeString;
    std::vector<int> deviceIds;
    bool useDepthFillingOption;
    bool writeProbabilityImages;

    // Declare the supported options.
    po::options_description options("options");
    options.add_options()
    ("help", "produce help message")
    ("version", "show version and exit")
    ("folderPrediction", po::value<std::string>(&folderPrediction)->default_value(""),
            "folder to output prediction images. leave it empty to suppress writing of prediction images")
    ("folderTesting", po::value<std::string>(&folderTesting)->required(), "folder with test images")
    ("treeFile", po::value<std::vector<std::string> >(&treeFiles)->required(), "serialized tree(s) (JSON)")
    ("maxDepth", po::value<int>(&maxDepth)->default_value(-1), "maximal depth of the tree used for prediction")
    ("histogramBias", po::value<double>(&histogramBias)->default_value(0.0), "histogram bias")
    ("numThreads", po::value<int>(&numThreads)->default_value(tbb::task_scheduler_init::default_num_threads()),
            "number of threads")
    ("mode", po::value<std::string>(&modeString)->default_value("gpu"), "mode: 'cpu' or 'gpu'")
    ("deviceId", po::value<std::vector<int> >(&deviceIds), "GPU device id (multiple occurrences possible)")
    ("profile", po::value<bool>(&profiling)->implicit_value(true)->default_value(false), "profiling")
    ("useDepthFilling", po::value<bool>(&useDepthFillingOption)->implicit_value(true),
            "whether to do simple depth filling")
    ("writeProbabilityImages", po::value<bool>(&writeProbabilityImages)->implicit_value(true)->default_value(false),
            "whether to write out probability PNGs")
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

    INFO("histogramBias: " << histogramBias);

    utils::Profile::setEnabled(profiling);

    tbb::task_scheduler_init init(numThreads);

    tbb::concurrent_vector<boost::shared_ptr<RandomTreeImage>> trees;
    tbb::concurrent_vector<TrainingConfiguration> configurations;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, treeFiles.size(), 1),
            [&](const tbb::blocked_range<size_t>& range) {

                for(size_t tree = range.begin(); tree != range.end(); tree++) {
                    INFO("reading tree " << tree << " from " << treeFiles[tree]);

                    boost::shared_ptr<RandomTreeImage> randomTree;

                    std::string hostname;
                    boost::filesystem::path folderTraining;
                    boost::posix_time::ptime date;

                    TrainingConfiguration configuration = RandomTreeImport::readJSON(treeFiles[tree], randomTree, hostname,
                            folderTraining, date);

                    INFO("trained " << date << " on " << hostname);
                    INFO("training folder: " << folderTraining);

                    assert(randomTree);
                    trees.push_back(randomTree);
                    configurations.push_back(configuration);

                    INFO(*randomTree);
                }

            });

    for (size_t i = 1; i < treeFiles.size(); i++) {
        bool strict = false;
        if (!configurations[0].equals(configurations[i], strict)) {
            ERROR("configuration of tree 0: " << configurations[0]);
            ERROR("configuration of tree " << i << ": " << configurations[i]);
            throw std::runtime_error("different configurations");
        }

        if (trees[0]->getTree()->getNumClasses() != trees[i]->getTree()->getNumClasses()) {
            ERROR("number of classes of tree 0: " << trees[0]->getTree()->getNumClasses());
            ERROR("number of classes of tree " << i << ": " << trees[i]->getTree()->getNumClasses());
            throw std::runtime_error("different number of classes in trees");
        }
    }

    INFO("training configuration " << configurations[0]);

    initDevice();

    TrainingConfiguration configuration = configurations[0];

    configuration.setDeviceIds(deviceIds);
    configuration.setAccelerationMode(TrainingConfiguration::parseAccelerationModeString(modeString));

    std::vector<boost::shared_ptr<RandomTreeImage>> treeVector;
    for (auto& tree : trees) {
        treeVector.push_back(tree);
    }
    assert(treeVector.size() == trees.size());

    RandomTreeImageEnsemble ensemble(treeVector, configuration);

    bool useDepthFilling = configuration.isUseDepthFilling();
    if (vm.count("useDepthFilling")) {
        if (useDepthFillingOption != useDepthFilling) {
            if (useDepthFilling) {
                WARNING("random forest was trained with depth filling. but prediction is performed without!");
            } else {
                WARNING(
                        "random forest was NOT trained with depth filling. but prediction is performed with depth filling!");
            }
        }
        // override
        useDepthFilling = useDepthFillingOption;
    }

    test(ensemble, folderTesting, folderPrediction, histogramBias, useDepthFilling, writeProbabilityImages, maxDepth);

    INFO("finished");
    return EXIT_SUCCESS;
}

