#pragma once

#include "embed/embed.h"

namespace brovisionml::api {

/// Mounts `bro.vision` onto `bro` in the current Bronze realm.
void installVision();

} // namespace brovisionml::api

using brovisionml::api::installVision;
