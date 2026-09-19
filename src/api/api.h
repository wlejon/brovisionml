#pragma once

#include "embed/embed.h"

#include <functional>
#include <string>

namespace brovisionml::api {

/// Mounts `bro.vision` onto `bro` in the current Bronze realm.
void installVision();

/// How a path handed to a `bro.vision` loader (`loadModel`, `loadDepth`,
/// `loadSam`, `loadNormal`, `loadBirefnet`, `loadStyleGAN3`, `loadDinov2`,
/// `loadDinov3`) or to an image argument given as a filename becomes a
/// filesystem path. Unset, the path is used as given; a host sets its `fs`
/// resolver so a relative path means what it means to the app
/// (bro: `setPathResolver(&brokit::api::resolveAssetPath)` before
/// installVision, the way it does for brotensor and brodiffusion).
/// Process-wide: every realm shares the host's filesystem.
void setPathResolver(std::function<std::string(const std::string&)> resolver);

} // namespace brovisionml::api

using brovisionml::api::installVision;
