// brovisionml::api::setPathResolver: every model path `bro.vision` takes
// goes through the host's resolver first, so a relative path means what it
// means to the app rather than to the process CWD.
//
// Mirrors brotensor's resolver: one process-global slot, unset by default.
// Nothing here builds a network or reads weights — resolution happens ahead
// of the loaders' "model dir not found" check, and that message names the
// path they actually looked at, which is the whole probe.

#include "../src/api/api.h"
#include "embed/embed.h"
#include "eval/eval.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace ev = bronze::embed;

std::filesystem::path g_scratch;
std::vector<std::string> g_asked;

void fail(const std::string& what) {
    std::cerr << "  path-resolver check failed: " << what << std::endl;
    std::exit(1);
}

std::string runScript(const char* label, const std::string& script) {
    ev::CallResult res = bronze::eval::evalScript(script);
    if (res.thrown) fail(std::string(label) + " threw: " + ev::toUtf8(res.value));
    return ev::toUtf8(res.value);
}

// "vvfs" and "vvfs/<name>" land in the scratch dir; anything else is left
// alone, the way a host resolver leaves an absolute path alone.
std::string resolve(const std::string& p) {
    g_asked.push_back(p);
    if (p == "vvfs") return g_scratch.string();
    if (p.rfind("vvfs/", 0) == 0) return (g_scratch / p.substr(5)).string();
    return p;
}

// Ask one loader for a model directory that is not there, and hand back the
// message. Every loader shares the check, so the message quotes whichever
// path it resolved to.
std::string loadMissing(const char* loader) {
    std::string script =
        std::string("(function() { try { bro.vision.") + loader +
        "(\"vvfs/missing\"); } catch (e) { return \"THREW:\" + e.message; } "
        "return \"NO-THROW\"; })()";
    return runScript(loader, script);
}

} // namespace

void brovisionmlTestPathResolver() {
    std::cout << "Checking brovisionml::api::setPathResolver..." << std::endl;

    g_scratch = std::filesystem::temp_directory_path() / "brovisionml_api_paths";
    std::filesystem::remove_all(g_scratch);
    std::filesystem::create_directories(g_scratch);
    const std::string resolved = (g_scratch / "missing").string();

    // 1. Unresolved: the loader looks at the path as given and says so.
    const std::string before = loadMissing("loadDepth");
    if (before.rfind("THREW:", 0) != 0 || before.find("vvfs/missing") == std::string::npos) {
        fail("loadDepth without a resolver: " + before);
    }
    if (!g_asked.empty()) fail("the resolver was consulted before it was installed");

    // 2. Installed, the same relative name resolves into the scratch dir —
    //    process-wide, not per realm.
    brovisionml::api::setPathResolver(&resolve);

    const char* kLoaders[] = {
        "loadModel", "loadDepth", "loadSam", "loadNormal",
        "loadBirefnet", "loadStyleGAN3", "loadDinov2", "loadDinov3",
    };
    for (const char* loader : kLoaders) {
        const size_t askedBefore = g_asked.size();
        const std::string out = loadMissing(loader);
        if (g_asked.size() != askedBefore + 1 || g_asked.back() != "vvfs/missing") {
            fail(std::string(loader) + " did not consult the resolver");
        }
        if (out.rfind("THREW:", 0) != 0) {
            fail(std::string(loader) + " on a missing model dir: " + out);
        }
        if (out.find("vvfs/missing") != std::string::npos) {
            fail(std::string(loader) + " still used the unresolved path: " + out);
        }
        if (out.find(resolved) == std::string::npos) {
            fail(std::string(loader) + " did not look at the resolved path: " + out);
        }
    }

    // 3. Clearing it restores take-it-as-given, so a host that never sets a
    //    resolver is unaffected.
    const size_t askedWithResolver = g_asked.size();
    brovisionml::api::setPathResolver(nullptr);
    const std::string cleared = loadMissing("loadDepth");
    if (cleared.rfind("THREW:", 0) != 0 || cleared.find("vvfs/missing") == std::string::npos) {
        fail("loadDepth still resolved after the resolver was cleared: " + cleared);
    }
    if (g_asked.size() != askedWithResolver) {
        fail("the resolver was still consulted after being cleared");
    }

    std::filesystem::remove_all(g_scratch);
    std::cout << "  brovisionml path resolver OK (" << askedWithResolver
              << " resolved paths across " << (sizeof(kLoaders) / sizeof(kLoaders[0]))
              << " loaders)." << std::endl;
}
