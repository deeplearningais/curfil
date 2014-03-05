#ifndef CURFIL_UTILS_H
#define CURFIL_UTILS_H

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/format.hpp>
#include <sstream>
#include <string>

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
