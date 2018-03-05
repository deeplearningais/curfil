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
#ifndef CURFIL_UTILS_H
#define CURFIL_UTILS_H

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/format.hpp>
#include <sstream>
#include <string>

#ifdef __CUDACC__
// results in a warning, though thrust seems to have gone that route in 
// https://github.com/jaredhoberock/thrust/commit/e13dbc444566ea8589d1c02e6df1c5a5533efb79
// using ::isnan;
#else
using std::isnan;
#endif

namespace curfil {

void logVersionInfo();

#define cudaSafeCall(X) X; curfil::utils::checkCudaError(#X);

// for debugging purposes
template<class T>
static inline void assert_equals(const T a, const T b) {
    if (a != b) {
        assert(a == b);
    }
}

#ifndef NDEBUG
#define assertProbability(probability) { \
    if (probability < 0.0 || probability > 1.0) { \
        printf("illegal probability: %lf\n", static_cast<double>(probability)); \
    } \
    assert(probability >= 0.0); \
    assert(probability <= 1.0); \
}
#else
#define assertProbability(probability) {}
#endif

namespace utils {

void checkCudaError(const char* msg);

/**
 * Used to get seconds or milliseconds passed for profiling purposes
 */
class Timer {
public:
    Timer() :
            isStopped(false) {
        start();
    }

    void reset(); /**< stop then start the timer */
    void start(); /**< start the timer */
    void stop(); /**< stop the timer */

    std::string format(int precision); /**< @return duration in seconds or milliseconds using the precision passed */

    double getSeconds(); /**< @return duration in seconds */
    double getMilliseconds(); /**< @return duration in milliseconds */

private:
    bool isStopped;
    boost::posix_time::ptime started;
    boost::posix_time::ptime stopped;

};

/**
 * A simple class used for averaging
 */
class Average {
public:
    Average() :
            sum(0), count(0) {
    }

    /**
     * add the value passed to the running sum and increments the counter
     */
    void addValue(const double& value) {
        sum += value;
        count++;
    }

    /**
     * @return the average of the values that were previously added
     */
    double getAverage() const {
        if (count == 0)
            return 0;
        return sum / count;
    }

private:
    double sum;
    size_t count;
};

void logMessage(const std::string& message, std::ostream& os);

#define CURFIL_LOG(level, message, os) { \
        std::ostringstream o; \
        o << boost::format("%-8s") % level; \
        o << message; \
        curfil::utils::logMessage(o.str(), os); \
    }

#define CURFIL_INFO(x) CURFIL_LOG("INFO", x, std::cout)

#define CURFIL_WARNING(x) CURFIL_LOG("WARNING", x, std::cout)

#define CURFIL_ERROR(x) CURFIL_LOG("ERROR", x, std::cout)

#ifdef CURFIL_DEBUG
#undef CURFIL_DEBUG
#endif

#ifdef NDEBUG
#define CURFIL_DEBUG(x) {}
#else
#define CURFIL_DEBUG(x) CURFIL_LOG("DEBUG", x, std::cout)
#endif

/**
 * Used to profile different stages of training and prediction
 */
class Profile {
public:
	/**
	 * @param name profile name
	 */
    Profile(const std::string& name) :
            name(name), timer() {
    }

    ~Profile() {
        if (isEnabled()) {
            timer.stop();
            CURFIL_INFO("PROFILING('" << name << "'): " << timer.format(3));
        }
    }

    /**
     * @return the timer duration in seconds
     */
    double getSeconds() {
        return timer.getSeconds();
    }

    /**
     * @return whether the profile is enabled
     */
    static bool isEnabled() {
        return enabled;
    }

    /**
     * enabling the profile
     */
    static void setEnabled(bool enable) {
        enabled = enable;
        CURFIL_INFO("profiling " << ((enabled) ? "enabled" : "disabled"));
    }

private:

    static bool enabled;

    std::string name;
    Timer timer;
};

size_t getFreeMemoryOnGPU(int deviceId);

}
}

#endif
