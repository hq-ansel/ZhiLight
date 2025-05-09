#pragma once
#include <sys/time.h>
#include <iostream>

namespace easyengine {
namespace logger {

static inline long get_time_us() {
    struct timeval now;
    gettimeofday(&now, NULL);
    return static_cast<long>(now.tv_sec * 1000 * 1000 + now.tv_usec);
}

} // namespace logger
} // namespace easyengine
