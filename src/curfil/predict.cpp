#include "predict.h"

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <iomanip>
#include <tbb/mutex.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_for.h>

#include "image.h"
#include "ndarray_ops.h"
#include "random_tree_image_ensemble.h"
#include "random_tree_image.h"
#include "utils.h"

namespace curfil {

void ConfusionMatrix::reset() {
    double* p = data.ptr();
    for (size_t i = 0; i < data.size(); i++) {
        p[i] = 0.0;
    }
}

void ConfusionMatrix::operator+=(const ConfusionMatrix& other) {
    if (other.getNumClasses() != getNumClasses()) {
        throw std::runtime_error("different number of classes");
    }

    data += other.data;
}

double ConfusionMatrix::averageClassAccuracy(bool includeVoid) const {
    utils::Average averageClassAccuracy;

    for (unsigned int label = 0; label < getNumClasses(); label++) {
        double classAccuracy = data(label, label);
        assertProbability(classAccuracy);
        if (includeVoid || label > 0) {
            averageClassAccuracy.addValue(classAccuracy);
        }
    }

    return averageClassAccuracy.getAverage();
}

void ConfusionMatrix::normalize() {

    cuv::ndarray<double, cuv::host_memory_space> sums(getNumClasses());

    const unsigned int numClasses = getNumClasses();

    for (unsigned int label = 0; label < numClasses; label++) {
        sums(label) = 0.0;
        for (unsigned int prediction = 0; prediction < numClasses; prediction++) {
            double value = static_cast<double>(data(label, prediction));
            sums(label) += value;
        }
    }

    for (unsigned int label = 0; label < numClasses; label++) {
        if (sums(label) == 0.0)
            continue;
        for (unsigned int prediction = 0; prediction < numClasses; prediction++) {
            data(label, prediction) /= sums(label);
        }
    }

#ifndef NDEBUG
    for (unsigned int label = 0; label < numClasses; label++) {
        double sum = 0.0;
        for (unsigned int prediction = 0; prediction < numClasses; prediction++) {
            double v = static_cast<double>(data(label, prediction));
            assert(v >= 0.0 && v <= 1.0);
            sum += v;
        }
        assert(sum == 0.0 || abs(1.0 - sum) < 1e-6);
    }
#endif

}

double calculateAccuracy(const LabelImage& image, const LabelImage* groundTruth,
        ConfusionMatrix& confusionMatrix) {
    assert(groundTruth != NULL);

    size_t correct = 0;
    confusionMatrix.reset();

    for (int y = 0; y < image.getHeight(); ++y) {
        for (int x = 0; x < image.getWidth(); ++x) {
            const LabelType label = groundTruth->getLabel(x, y);
            const LabelType prediction = image.getLabel(x, y);

            if (prediction == label) {
                correct++;
            }

            confusionMatrix.increment(label, prediction);
        }
    }

    return static_cast<double>(correct) / (image.getWidth() * image.getHeight());
}

double calculateAccuracyNoBackground(const LabelImage& image, const LabelImage* groundTruth) {
    assert(groundTruth != NULL);
    size_t truePositives = 0;
    size_t trueNegatives = 0;
    size_t falseNegatives = 0;
    size_t falsePositives = 0;

    for (int y = 0; y < image.getHeight(); ++y) {
        for (int x = 0; x < image.getWidth(); ++x) {
            const LabelType label = groundTruth->getLabel(x, y);
            const LabelType prediction = image.getLabel(x, y);

            if (prediction == label) {
                if (prediction == 0) {
                    trueNegatives++;
                } else {
                    truePositives++;
                }
            } else {
                if (prediction == 0) {
                    falseNegatives++;
                } else {
                    falsePositives++;
                }
            }
        }
    }

    return static_cast<double>(truePositives) / (truePositives + falseNegatives + falsePositives);
}

double calculateAccuracyNoVoid(const LabelImage& image, const LabelImage* groundTruth) {
    assert(groundTruth != NULL);
    size_t correct = 0;
    size_t wrong = 0;

    for (int y = 0; y < image.getHeight(); ++y) {
        for (int x = 0; x < image.getWidth(); ++x) {
            const LabelType label = groundTruth->getLabel(x, y);

            if (label == 0) {
                // skip void
                continue;
            }

            const LabelType prediction = image.getLabel(x, y);
            if (prediction == label) {
                correct++;
            } else {
                wrong++;
            }
        }
    }

    return static_cast<double>(correct) / (correct + wrong);
}

void test(RandomTreeImageEnsemble& randomForest, const std::string& folderTesting,
        const std::string& folderPrediction, const double histogramBias, const bool useDepthFilling,
        const bool writeProbabilityImages, const int maxDepth) {

    auto filenames = listImageFilenames(folderTesting);
    if (filenames.empty()) {
        throw std::runtime_error(std::string("found no files in ") + folderTesting);
    }

    INFO("got " << filenames.size() << " files for prediction");

    INFO("label/color map:");
    const auto labelColorMap = randomForest.getLabelColorMap();
    for (const auto& labelColor : labelColorMap) {
        const auto color = LabelImage::decodeLabel(labelColor.first);
        INFO("label: " << static_cast<int>(labelColor.first) << ", color: RGB(" << color << ")");
    }

    if (maxDepth > 0) {
        throw std::runtime_error("setting maxDepth is currently not implemented");
    }

    tbb::mutex totalMutex;
    utils::Average averageAccuracy;
    utils::Average averageAccuracyNoBackground;
    utils::Average averageAccuracyNoVoid;

    const LabelType numClasses = randomForest.getNumClasses();
    ConfusionMatrix totalConfusionMatrix(numClasses);

    size_t i = 0;

    const bool useCIELab = randomForest.getConfiguration().isUseCIELab();
    INFO("CIELab: " << useCIELab);
    INFO("DepthFilling: " << useDepthFilling);

    randomForest.normalizeHistograms(histogramBias);

    bool onGPU = randomForest.getConfiguration().getAccelerationMode() == GPU_ONLY;

    size_t grainSize = 1;
    if (!onGPU) {
        grainSize = filenames.size();
    }

    bool writeImages = true;
    if (folderPrediction.empty()) {
        WARNING("no prediction folder given. will not write images");
        writeImages = false;
    }

    tbb::parallel_for(tbb::blocked_range<size_t>(0, filenames.size(), grainSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for(size_t fileNr = range.begin(); fileNr != range.end(); fileNr++) {
                    const std::string& filename = filenames[fileNr];
                    const auto imageLabelPair = loadImagePair(filename, useCIELab, useDepthFilling);
                    const RGBDImage* testImage = imageLabelPair.getRGBDImage();
                    const LabelImage* groundTruth = imageLabelPair.getLabelImage();
                    LabelImage prediction(testImage->getWidth(), testImage->getHeight());

                    for(int y = 0; y < groundTruth->getHeight(); y++) {
                        for(int x = 0; x < groundTruth->getWidth(); x++) {
                            const LabelType label = groundTruth->getLabel(x, y);
                            if (label >= numClasses) {
                                const auto msg = (boost::format("illegal label in ground truth image '%s' at pixel (%d,%d): %d RGB(%3d,%3d,%3d) (numClasses: %d)")
                                        % filename
                                        % x % y
                                        % static_cast<int>(label)
                                        % LabelImage::decodeLabel(label)[0]
                                        % LabelImage::decodeLabel(label)[1]
                                        % LabelImage::decodeLabel(label)[2]
                                        % static_cast<int>(numClasses)
                                ).str();
                                throw std::runtime_error(msg);
                            }
                        }
                    }

                    boost::filesystem::path fn(testImage->getFilename());
                    const std::string basepath = folderPrediction + "/" + boost::filesystem::basename(fn);

                    const cuv::ndarray<float, cuv::host_memory_space> probabilities = randomForest.test(testImage, prediction, onGPU);

#ifndef NDEBUG
            for(LabelType label = 0; label < randomForest.getNumClasses(); label++) {
                if (!randomForest.shouldIgnoreLabel(label)) {
                    continue;
                }

                // ignored classes must not be predicted as we did not sample them
                for(size_t y = 0; y < probabilities.shape(0); y++) {
                    for(size_t x = 0; x < probabilities.shape(1); x++) {
                        const float& probability = probabilities(label, y, x);
                        assert(probability == 0.0);
                    }
                }
            }
#endif

            if (writeImages && writeProbabilityImages) {
                utils::Profile profile("writeProbabilityImages");
                RGBDImage probabilityImage(testImage->getWidth(), testImage->getHeight());
                for(LabelType label = 0; label< randomForest.getNumClasses(); label++) {

                    if (!randomForest.shouldIgnoreLabel(label)) {
                        continue;
                    }

                    for(int y = 0; y < probabilityImage.getHeight(); y++) {
                        for(int x = 0; x < probabilityImage.getWidth(); x++) {
                            const float& probability = probabilities(label, y, x);
                            for(int c=0; c<3; c++) {
                                probabilityImage.setColor(x, y, c, probability);
                            }
                        }
                    }
                    const std::string filename = boost::str(boost::format("%s_label_%d.png") % basepath % static_cast<int>(label));
                    probabilityImage.writeColor(filename);
                }
            }

            int thisNumber;

            {
                tbb::mutex::scoped_lock total(totalMutex);
                thisNumber = i++;
            }

            if (writeImages) {
                utils::Profile profile("writeImages");
                testImage->writeColor(basepath + ".png");
                testImage->writeDepth(basepath + "_depth.png");
                groundTruth->write(basepath + "_ground_truth.png");
                prediction.write(basepath + "_prediction.png");
            }

            ConfusionMatrix confusionMatrix(numClasses);

            double accuracy = calculateAccuracy(prediction, groundTruth, confusionMatrix);
            double accuracyNoBackground = calculateAccuracyNoBackground(prediction, groundTruth);
            double accuracyNoVoid = calculateAccuracyNoVoid(prediction, groundTruth);

            tbb::mutex::scoped_lock lock(totalMutex);

            INFO("prediction " << (thisNumber+1) << "/" << filenames.size() << " (" << testImage->getFilename() << "): pixel accuracy, without background, no void: " << 100 * accuracy
                    << ", " << 100 * accuracyNoBackground << ", " << 100 * accuracyNoVoid);

            averageAccuracy.addValue(accuracy);
            averageAccuracyNoBackground.addValue(accuracyNoBackground);
            averageAccuracyNoVoid.addValue(accuracyNoVoid);

            totalConfusionMatrix += confusionMatrix;

            accuracy = averageAccuracy.getAverage();
            accuracyNoBackground = averageAccuracyNoBackground.getAverage();
            accuracyNoVoid = averageAccuracyNoVoid.getAverage();
            INFO("average accuracy: " << 100 * accuracy);
            INFO("average accuracy without background: " << 100 * accuracyNoBackground);
            INFO("average accuracy no void: " << 100 * accuracyNoVoid);
        }

    });

    tbb::mutex::scoped_lock lock(totalMutex);
    double accuracy = averageAccuracy.getAverage();
    double accuracyNoBackground = averageAccuracyNoBackground.getAverage();
    double accuracyNoVoid = averageAccuracyNoVoid.getAverage();

    totalConfusionMatrix.normalize();

    INFO(totalConfusionMatrix);

    INFO("total accuracy: " << 100 * accuracy);
    INFO("total accuracy without background: " << 100 * accuracyNoBackground);
    INFO("total accuracy no void: " << 100 * accuracyNoVoid);
}

}

std::ostream& operator<<(std::ostream& o, const curfil::ConfusionMatrix& confusionMatrix) {
    const unsigned int numClasses = confusionMatrix.getNumClasses();

    o << numClasses << "x" << numClasses << " confusion matrix (y: labels, x: predictions):" << std::endl;

    for (curfil::LabelType label = 0; label < numClasses; label++) {

        const curfil::RGBColor color = curfil::LabelImage::decodeLabel(label);

        o << " class " << static_cast<int>(label) << " RGB(" << std::setw(3 * 3 + 2) << color << std::flush << ") : ";

        for (curfil::LabelType prediction = 0; prediction < numClasses; prediction++) {
            double probability = static_cast<double>(confusionMatrix(label, prediction));
            o << (boost::format("%7.3f") % probability).str();
        }
        o << std::endl;
    }

    o << std::endl;

    o << "average class accuracy (incl void): " << 100 * confusionMatrix.averageClassAccuracy(true) << std::endl;
    o << "average class accuracy (no void): " << 100 * confusionMatrix.averageClassAccuracy(false) << std::endl;

    return o;
}
