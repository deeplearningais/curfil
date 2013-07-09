#include "utils.h"

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <cuda_runtime_api.h>

#include "version.h"

namespace curfil {

void logVersionInfo() {
#ifdef NDEBUG
    INFO("release version " << getVersion());
#else
    WARNING("this is the debugging version " << getVersion());
#endif
}

namespace utils {

bool Profile::enabled = true;

void Timer::reset() {
    stop();
    start();
}

void Timer::start() {
    started = boost::posix_time::microsec_clock::local_time();
    isStopped = false;
}
void Timer::stop() {
    stopped = boost::posix_time::microsec_clock::local_time();
    isStopped = true;
}

std::string Timer::format(int precision) {
    std::ostringstream o;
    o.precision(precision);
    if (getSeconds() <= 1.0) {
        o << std::fixed << getMilliseconds() << " ms";
    } else {
        o << std::fixed << getSeconds() << " s";
    }
    return o.str();
}

double Timer::getSeconds() {
    if (!isStopped)
        stop();
    boost::posix_time::time_duration duration = (stopped - started);
    return duration.total_microseconds() / static_cast<double>(1e6);
}

double Timer::getMilliseconds() {
    if (!isStopped)
        stop();
    boost::posix_time::time_duration duration = (stopped - started);
    return duration.total_microseconds() / static_cast<double>(1e3);
}

void logMessage(const std::string& msg, std::ostream& os) {
    boost::posix_time::ptime date_time = boost::posix_time::microsec_clock::local_time();

    std::string out = boost::str(boost::format("%s  %s") % date_time % msg);
    std::ostringstream endlStream;
    endlStream << std::endl;
    std::string endl = endlStream.str();

    // append newline if it’s not already there
    if (!boost::ends_with(out, endl)) {
        out += endl;
    }
    os << out << std::flush;
}

void checkCudaError(const char* msg) {
    cudaError_t lastError = cudaGetLastError();
    if (lastError == cudaSuccess)
        return;

    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(lastError));
}

}
}
