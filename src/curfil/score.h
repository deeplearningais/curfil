#ifndef CURFIL_SCORE_H
#define CURFIL_SCORE_H

#include <assert.h>
#include <cuv/ndarray.hpp>

#include "utils.h"

namespace curfil
{

typedef double ScoreType;

// Normalized information gain score, see formula (11) in [Wehenkel1991] and
// appendix A in [Geurts2006].
class InformationGainScore {

protected:
    __host__ __device__
    static ScoreType entropy(const ScoreType prob) {
        if (prob == 0.0) {
            return 0.0;
        }
        return -prob * log2(prob);
    }

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

class NormalizedInformationGainScore: public InformationGainScore {
protected:

    // H_s: "split entropy", i.e. a measure of the split balancedness
    __host__ __device__
    static ScoreType splitEntropy(const ScoreType total, const ScoreType totalLeft, const ScoreType totalRight) {
        ScoreType H_s = (entropy(totalLeft) + entropy(totalRight) - entropy(total)) / total;

        assert(!isnan(H_s));
        assert(H_s >= 0);

        return H_s;
    }

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

}

#endif
