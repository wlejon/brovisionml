// profile.h — env-gated stage profiler shared by every model's forward path.
//
// Set BROVISIONML_PROFILE=1 to print sequential interval stamps: each
// profile_mark(dev, "name") syncs the device (so async backends attribute
// cost to the stage that actually ran) and prints the wall time since the
// previous mark on this thread. A nullptr name resets the interval origin
// without printing — call it at the top of a forward. When the env var is
// unset, marks cost one branch.
//
// Internal header (src/), not installed.

#pragma once

#include "brotensor/runtime.h"

namespace brovisionml::detail {

bool profile_enabled();
void profile_mark(brotensor::Device dev, const char* name);

}  // namespace brovisionml::detail
