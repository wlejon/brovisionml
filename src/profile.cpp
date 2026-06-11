#include "profile.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace brovisionml::detail {

bool profile_enabled() {
    static const bool on = []() {
        const char* v = std::getenv("BROVISIONML_PROFILE");
        return v && v[0] && v[0] != '0';
    }();
    return on;
}

void profile_mark(brotensor::Device dev, const char* name) {
    if (!profile_enabled()) return;
    brotensor::sync(dev);
    static thread_local std::chrono::steady_clock::time_point last =
        std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (name) {
        const double ms =
            std::chrono::duration<double, std::milli>(now - last).count();
        std::fprintf(stderr, "[brovisionml-prof] %-28s %9.2f ms\n", name, ms);
    }
    last = now;
}

}  // namespace brovisionml::detail
