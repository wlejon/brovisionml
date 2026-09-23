#include "api.h"
#include "host_vision_internal.h"

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
