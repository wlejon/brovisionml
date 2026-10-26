#include "api.h"
#include "host_vision_internal.h"
#include "embed/embed_access.h"

#include <filesystem>
#include <string_view>
#include <system_error>

namespace brovisionml::api {

namespace {

std::function<std::string(const std::string&)>& pathResolverSlot() {
    static std::function<std::string(const std::string&)> resolver;
    return resolver;
}

} // namespace

std::string resolvePath(const std::string& path) {
    auto& r = pathResolverSlot();
    return r ? r(path) : path;
}

PathRefusal checkReadAccess(std::string_view path) {
    if (path.empty() || !ev::accessGateActive()) return {};
    std::filesystem::path p(std::u8string(path.begin(), path.end()));
    std::error_code ec;
    std::filesystem::path abs = std::filesystem::absolute(p, ec);
    if (ec) abs = p;
    const std::u8string u = abs.lexically_normal().u8string();
    PathRefusal r;
    r.path.assign(u.begin(), u.end());
    r.message = ev::checkAccess(ev::AccessKind::Read, r.path);
    return r;
}

Value throwRefusal(const PathRefusal& refusal) {
    return ev::throwAccessError(refusal.message, "EACCES", refusal.path);
}

void setPathResolver(std::function<std::string(const std::string&)> resolver) {
    pathResolverSlot() = std::move(resolver);
}

void installVision() {
    ensureVisionClassesInstalled();

    // Everything below allocates (getProperty, createObject, setProperty), so
    // both roots live in Persistents and are re-read at every use.
    ev::Persistent globalThisP(ev::undefined());
    {
        auto gt = ev::globalValue("globalThis");
        if (gt.found && ev::isObject(gt.value)) globalThisP.set(gt.value);
    }

    ev::Persistent broP(ev::undefined());
    {
        auto g = ev::globalValue("bro");
        if (g.found && ev::isObject(g.value)) broP.set(g.value);
    }
    if (!ev::isObject(broP.get()) && ev::isObject(globalThisP.get())) {
        Value candidate = ev::getProperty(globalThisP.get(), "bro");
        if (ev::isObject(candidate)) broP.set(candidate);
    }
    if (!ev::isObject(broP.get())) {
        broP.set(ev::createObject());
        ev::registerGlobal("bro", broP.get());
        if (ev::isObject(globalThisP.get())) {
            globalThisP.set(ev::setProperty(globalThisP.get(), "bro", broP.get()));
        }
    }

    // Mount bro.vision
    ev::Persistent visionP(makeVisionNamespace());
    broP.set(ev::setProperty(broP.get(), "vision", visionP.get()));
}

} // namespace brovisionml::api
