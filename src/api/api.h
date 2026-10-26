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

// ---- Access ------------------------------------------------------------------
//
// Every native that opens a path the app named asks the host's access gate
// (embed/embed_access.h, bronze::embed::checkAccess) first, on the calling JS
// thread, with AccessKind::Read and the resolved path made absolute and
// lexically normal (UTF-8), before anything is opened or stat'ed:
//   every loader (loadModel, loadDepth, loadSam, loadSam2, loadNormal,
//   loadHed, loadLineart, loadMlsd, loadOpenpose, loadSegformer,
//   loadBirefnet, loadStyleGAN3, loadDinov2, loadDinov3, loadOwlv2) for its
//   model path (a directory covers what is in it), and every method taking an
//   image as a filename (estimate, detect, segment, setImage, addFrame,
//   encode, invert, ...) for that file.
// Nothing in bro.vision writes a file.
//
// A refusal is thrown from the call as an Error carrying the gate's sentence as
// its message, code "EACCES" and `path` (the path as the gate was handed it),
// named "PermissionRevoked" for a grant taken back while the app ran. Nothing
// is read.

} // namespace brovisionml::api

using brovisionml::api::installVision;
