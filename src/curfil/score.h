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
#ifndef CURFIL_SCORE_H
#define CURFIL_SCORE_H

#include <assert.h>
#include <cuv/ndarray.hpp>

#include "utils.h"

namespace curfil
{

typedef double ScoreType;

/**
 * Information Gain calculation using Shannon entropy of the parent and child nodes
 * @ingroup score_calc
 */
class InformationGainScore {

protected:

    /**
     * @return entropy calculated using the given probability
     */
    __host__ __device__
    static ScoreType entropy(const ScoreType prob) {
        if (prob == 0.0) {
            return 0.0;
        }
        return -prob * log2(prob);
    }

    /**
     * @return score in the interval [0,1]
     */
    __host__ __device__
    static ScoreType normalizeScore(const ScoreType score) {

        assert(!isnan(score));

        if (score > 1.0) {
            assert(fabs(score - 1) < 1e-6);
            return 1;
        }

        if (score < 0.0) {
            assert(fabs(score - 0) < 1e-6);
            return 0;
        }

        assertProbability(score);
        return score;
    }

public:

    /**
     * @return normalized score calculated after splitting
     */
    template<class W>
    __host__ __device__
    static ScoreType calculateScore(const size_t numLabels, const W* leftClasses, const W* rightClasses,
            const unsigned int leftRightStride, const W* allClasses, const ScoreType totalLeft,
            const ScoreType totalRight) {

        const ScoreType total = totalLeft + totalRight;

        const ScoreType leftProb = totalLeft / total;
        const ScoreType rightProb = totalRight / total;

#ifndef NDEBUG
        W totalLeftTest = 0;
        W totalRightTest = 0;
#endif

        ScoreType score = 0;

        for (size_t label = 0; label < numLabels; label++) {
            const size_t offset = label * leftRightStride;
            const W& leftValue = leftClasses[offset];
            const W& rightValue = rightClasses[offset];

#ifndef NDEBUG
            assert(leftValue <= allClasses[label]);
            assert(rightValue <= allClasses[label]);
            assert(leftValue <= total);
            assert(rightValue <= total);
            totalLeftTest += leftValue;
            totalRightTest += rightValue;
#endif

           const ScoreType classProb = allClasses[label] / total;
            assertProbability(classProb);

            if (leftValue > 0) {
                ScoreType classProbLeft = leftValue / total;
                assertProbability(classProbLeft);
                score += classProbLeft * log2(classProbLeft / (leftProb * classProb));
            }

            if (rightValue > 0) {
                ScoreType classProbRight = rightValue / total;
                assertProbability(classProbRight);
                score += classProbRight * log2(classProbRight / (rightProb * classProb));
            }
        }

        assert(totalLeftTest == totalLeft);
        assert(totalRightTest == totalRight);

        return normalizeScore(score);
    }
};

/**
 * Information gain normalized by the sum of the classification and split entropies 
 * see formula (11) in [Wehenkel1991] and appendix A in [Geurts2006]
 * @ingroup score_calc
 */ 
class NormalizedInformationGainScore: public InformationGainScore {
protected:

    /**
     * @return H_s: "split entropy", i.e. a measure of the split balancedness
     */
    __host__ __device__
    static ScoreType splitEntropy(const ScoreType total, const ScoreType totalLeft, const ScoreType totalRight) {
        ScoreType H_s = (entropy(totalLeft) + entropy(totalRight) - entropy(total)) / total;

        assert(!isnan(H_s));
        assert(H_s >= 0);

        return H_s;
    }

    /**
     * @return classification entropy
     */
    template<class W>
    __host__ __device__
    static ScoreType classificationEntropy(const size_t numLabels, const W* allClasses, const ScoreType total) {

        ScoreType H_c = 0;

        for (size_t label = 0; label < numLabels; label++) {
            const W& value = allClasses[label];
            assert(value <= total);
            if (value > 0) {
                H_c += entropy(value);
            }
        }

        H_c -= entropy(total);
        H_c /= total;

        assert(!isnan(H_c));
        assert(H_c >= 0);

        return H_c;
    }

public:

    template<class W>
    __host__ __device__
    static ScoreType calculateScore(const size_t numClasses, const W* leftClasses, const W* rightClasses,
            const unsigned int leftRightStride, const W* allClasses, const ScoreType totalLeft,
            const ScoreType totalRight) {

        // Compute information gain due to split decision
        const ScoreType informationGain = InformationGainScore::calculateScore(numClasses,
                leftClasses, rightClasses, leftRightStride,
                allClasses, totalLeft, totalRight);

        if (informationGain == 0) {
            // skip calculation of split entropy
            return 0;
        }

        ScoreType total = totalLeft + totalRight;
        assert(total > 0);

        const ScoreType H_s = splitEntropy(total, totalLeft, totalRight);
        const ScoreType H_c = classificationEntropy(numClasses, allClasses, total);

        ScoreType score = (2 * informationGain) / (H_s + H_c);
        return normalizeScore(score);
    }
};

/// @cond DEV
class NoOpScore: public InformationGainScore {

public:
    template<class W>
    __host__ __device__
    static ScoreType calculateScore(const size_t numLabels, const W* leftClasses, const W* rightClasses,
            const unsigned int leftRightStride, const W* allClasses, const ScoreType totalLeft,
            const ScoreType totalRight) {

        const ScoreType total = totalLeft + totalRight;

        ScoreType score = 0;

        for (size_t label = 0; label < numLabels; label++) {
            const size_t offset = label * leftRightStride;
            const W& leftValue = leftClasses[offset];
            const W& rightValue = rightClasses[offset];

            score += leftValue;
            score += rightValue;
            score += allClasses[label];
        }

        return score / (3.0 * total);
    }
};

/// @endcond

}

#endif
