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
#include "predict.h"

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <iomanip>
#include <tbb/mutex.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_for.h>

#include "image.h"
#include "ndarray_ops.h"
#include "random_forest_image.h"
#include "random_tree_image.h"
#include "utils.h"

namespace curfil {

void ConfusionMatrix::reset() {
    double* p = data.ptr();
    for (size_t i = 0; i < data.size(); i++) {
        p[i] = 0.0;
    }
    normalized = false;
}

void ConfusionMatrix::resize(unsigned int numClasses) {
    data.resize(numClasses, numClasses);
    reset();
}

void ConfusionMatrix::operator+=(const ConfusionMatrix& other) {

    if (normalized) {
        throw std::runtime_error("confusion matrix is already normalized");
    }

    if (other.getNumClasses() != getNumClasses()) {
        std::ostringstream o;
        o << "different number of classes in confusion matrix: " << this->getNumClasses() << " and "
                << other.getNumClasses();
        throw std::runtime_error(o.str());
    }

    data += other.data;
}

double ConfusionMatrix::averageClassAccuracy(bool includeVoid) const {

    if (!normalized) {
        ConfusionMatrix normalizedConfusionMatrix(*this);
        normalizedConfusionMatrix.normalize();
        assert(normalizedConfusionMatrix.isNormalized());
        return normalizedConfusionMatrix.averageClassAccuracy(includeVoid);
    }

    utils::Average averageClassAccuracy;

    for (unsigned int label = 0; label < getNumClasses(); label++) {
        double classAccuracy = data(label, label);
        assertProbability(classAccuracy);
 
        bool ignore = false;
        if (!includeVoid && !ignoredLabels.empty())
        	for (LabelType ID: ignoredLabels)
        		if (ID == label)
       			{
       				ignore = true;
       				break;
       			}
        if (ignore)
        	continue;
        else
        	averageClassAccuracy.addValue(classAccuracy);
    }

    return averageClassAccuracy.getAverage();
}

void ConfusionMatrix::normalize() {

    if (normalized) {
        throw std::runtime_error("confusion matrix is already normalized");
    }

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

    normalized = true;

}

double calculatePixelAccuracy(const LabelImage& prediction, const LabelImage& groundTruth,
        const bool includeVoid, const std::vector<LabelType>* ignoredLabels,  ConfusionMatrix* confusionMatrix) {

    size_t correct = 0;
    size_t wrong = 0;

    if (confusionMatrix) {
        LabelType numClasses = 0;
        for (int y = 0; y < groundTruth.getHeight(); ++y) {
            for (int x = 0; x < groundTruth.getWidth(); ++x) {
                numClasses = std::max(numClasses, groundTruth.getLabel(x, y));
            }
        }
        numClasses++;
        if (confusionMatrix->getNumClasses() < numClasses) {
            confusionMatrix->resize(numClasses);
            confusionMatrix->reset();
        }
    }

    for (int y = 0; y < prediction.getHeight(); ++y) {
        for (int x = 0; x < prediction.getWidth(); ++x) {
            const LabelType label = groundTruth.getLabel(x, y);

            bool ignore = false;
	    // assert(!includeVoid && ignoredLabels);
            if (!includeVoid && !ignoredLabels->empty())
            	for (LabelType ID: *ignoredLabels)
            		if (ID == label)
           			{
           				ignore = true;
           				break;
           			}
            if (ignore)
            	continue;


            const LabelType predictedClass = prediction.getLabel(x, y);

            if (predictedClass == label) {
                correct++;
            } else {
                wrong++;
            }

            if (confusionMatrix) {
                confusionMatrix->increment(label, predictedClass);
            }
        }
    }

    size_t numPixels;

    if (includeVoid) {
        numPixels = prediction.getWidth() * prediction.getHeight();
    } else {
        numPixels = correct + wrong;
    }
    return static_cast<double>(correct) / numPixels;
}

void test(RandomForestImage& randomForest, const std::string& folderTesting,
        const std::string& folderPrediction, const bool useDepthFilling,
        const bool writeProbabilityImages) {

    auto filenames = listImageFilenames(folderTesting);
    if (filenames.empty()) {
        throw std::runtime_error(std::string("found no files in ") + folderTesting);
    }

    CURFIL_INFO("got " << filenames.size() << " files for prediction");

    CURFIL_INFO("label/color map:");
    const auto labelColorMap = randomForest.getLabelColorMap();
    for (const auto& labelColor : labelColorMap) {
        const auto color = LabelImage::decodeLabel(labelColor.first);
        CURFIL_INFO("label: " << static_cast<int>(labelColor.first) << ", color: RGB(" << color << ")");
    }

    tbb::mutex totalMutex;
    utils::Average averageAccuracy;
    utils::Average averageAccuracyWithoutVoid;

    size_t i = 0;

    const bool useCIELab = randomForest.getConfiguration().isUseCIELab();
    CURFIL_INFO("CIELab: " << useCIELab);
    CURFIL_INFO("DepthFilling: " << useDepthFilling);

    const bool useDepthImages = randomForest.getConfiguration().isUseDepthImages();
    CURFIL_INFO("useDepthImages: " << useDepthImages);

    bool onGPU = randomForest.getConfiguration().getAccelerationMode() == GPU_ONLY;

    size_t grainSize = 1;
    if (!onGPU) {
        grainSize = filenames.size();
    }

    bool writeImages = true;
    if (folderPrediction.empty()) {
        CURFIL_WARNING("no prediction folder given. will not write images");
        writeImages = false;
    }

    std::vector<LabelType> ignoredLabels;
    for (const std::string colorString : randomForest.getConfiguration().getIgnoredColors()) {
    	ignoredLabels.push_back(LabelImage::encodeColor(RGBColor(colorString)));
    }

    const LabelType numClasses = randomForest.getNumClasses();
    ConfusionMatrix totalConfusionMatrix(numClasses, ignoredLabels);
    double totalPredictionTime = 0;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, filenames.size(), grainSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for(size_t fileNr = range.begin(); fileNr != range.end(); fileNr++) {
                    const std::string& filename = filenames[fileNr];
                    const auto imageLabelPair = loadImagePair(filename, useCIELab, useDepthImages, useDepthFilling);
                    const RGBDImage& testImage = imageLabelPair.getRGBDImage();
                    const LabelImage& groundTruth = imageLabelPair.getLabelImage();
                    LabelImage prediction(testImage.getWidth(), testImage.getHeight());

                    for(int y = 0; y < groundTruth.getHeight(); y++) {
                        for(int x = 0; x < groundTruth.getWidth(); x++) {
                            const LabelType label = groundTruth.getLabel(x, y);
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

                    boost::filesystem::path fn(testImage.getFilename());
                    const std::string basepath = folderPrediction + "/" + boost::filesystem::basename(fn);

                    cuv::ndarray<float, cuv::host_memory_space> probabilities;
                    utils::Timer timer2;
                    prediction = randomForest.predict(testImage, &probabilities, onGPU, useDepthImages);
                    totalPredictionTime += timer2.getMilliseconds();

#ifndef NDEBUG
            for(LabelType label = 0; label < randomForest.getNumClasses(); label++) {
                if (!randomForest.shouldIgnoreLabel(label)) {
                    continue;
                }

                // ignored classes must not be predicted as we did not sample them
                for(size_t y = 0; y < probabilities.shape(1); y++) {
                    for(size_t x = 0; x < probabilities.shape(2); x++) {
                        const float& probability = probabilities(label, y, x);
                        assert(probability == 0.0);
                    }
                }
            }
#endif

            if (writeImages && writeProbabilityImages) {
                utils::Profile profile("writeProbabilityImages");
                RGBDImage probabilityImage(testImage.getWidth(), testImage.getHeight());
                for(LabelType label = 0; label< randomForest.getNumClasses(); label++) {

                    if (randomForest.shouldIgnoreLabel(label)) {
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
                    const std::string filename = (boost::format("%s_label_%d.png") % basepath % static_cast<int>(label)).str();
                    probabilityImage.saveColor(filename);
                }
            }

            int thisNumber;

            {
                tbb::mutex::scoped_lock total(totalMutex);
                thisNumber = i++;
            }

            if (writeImages) {
                utils::Profile profile("writeImages");
                testImage.saveColor(basepath + ".png");
                if (useDepthImages){
                	testImage.saveDepth(basepath + "_depth.png");}
                groundTruth.save(basepath + "_ground_truth.png");
                prediction.save(basepath + "_prediction.png");
            }

            ConfusionMatrix confusionMatrix(numClasses);
            double accuracy = calculatePixelAccuracy(prediction, groundTruth, true, &ignoredLabels);
            double accuracyWithoutVoid = calculatePixelAccuracy(prediction, groundTruth, false,  &ignoredLabels, &confusionMatrix);

            tbb::mutex::scoped_lock lock(totalMutex);

            CURFIL_INFO("prediction " << (thisNumber + 1) << "/" << filenames.size()
                    << " (" << testImage.getFilename() << "): pixel accuracy (without void): " << 100 * accuracy
                    << " (" << 100 * accuracyWithoutVoid << ")");

            averageAccuracy.addValue(accuracy);
            averageAccuracyWithoutVoid.addValue(accuracyWithoutVoid);

            totalConfusionMatrix += confusionMatrix;
        }

    });
    CURFIL_INFO(totalPredictionTime / filenames.size()<<" ms/image");

    tbb::mutex::scoped_lock lock(totalMutex);
    double accuracy = averageAccuracy.getAverage();
    double accuracyWithoutVoid = averageAccuracyWithoutVoid.getAverage();

    CURFIL_INFO(totalConfusionMatrix);

    CURFIL_INFO("pixel accuracy: " << 100 * accuracy);
    CURFIL_INFO("pixel accuracy without void: " << 100 * accuracyWithoutVoid);
}

}

std::ostream& operator<<(std::ostream& o, const curfil::ConfusionMatrix& confusionMatrix) {
    const unsigned int numClasses = confusionMatrix.getNumClasses();

    curfil::ConfusionMatrix normalizedConfusionMatrix(confusionMatrix);
    normalizedConfusionMatrix.normalize();

    assert(numClasses == normalizedConfusionMatrix.getNumClasses());

    o << numClasses << "x" << numClasses << " confusion matrix (y: labels, x: predictions):" << std::endl;

    o << "                               ";

    for (curfil::LabelType label = 0; label < numClasses; label++) {
        const curfil::RGBColor color = curfil::LabelImage::decodeLabel(label);
        o << "cl " << std::left << std::setw(2) << static_cast<int>(label) << "  ";
    }

    o << std::endl;

    for (curfil::LabelType label = 0; label < numClasses; label++) {

        const curfil::RGBColor color = curfil::LabelImage::decodeLabel(label);

        o << " class " << std::setw(2) << static_cast<int>(label) << " RGB(" << std::setw(3 * 3 + 2) << color << std::flush << ") : ";

        for (curfil::LabelType prediction = 0; prediction < numClasses; prediction++) {
            double probability = static_cast<double>(normalizedConfusionMatrix(label, prediction));
            o << (boost::format("%7.3f") % probability).str();
        }
        o << std::endl;
    }

    o << std::endl;

    o << "average class accuracy (incl void): " << 100 * normalizedConfusionMatrix.averageClassAccuracy(true) << std::endl;
    o << "average class accuracy (no void): " << 100 * normalizedConfusionMatrix.averageClassAccuracy(false) << std::endl;

    return o;
}
