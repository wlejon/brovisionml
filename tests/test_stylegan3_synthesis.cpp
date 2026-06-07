// StyleGAN3 SynthesisNetwork schedule/assembly test.
//
// The band-limit schedule (cutoffs/stopbands/sampling-rates/sizes/channels) and
// the layer naming are what a real checkpoint is keyed off, and a single wrong
// constant silently breaks loading. With no checkpoint we assert the structural
// facts the schedule must produce for the released config-R presets: the input
// map is always low-resolution, the layer count is num_layers+1, the first
// hidden layer is 36x36x1024, ToRGB emits img_channels at full resolution, and
// the layer names match the reference's L{idx}_{size}_{channels} format.
//
// Exact numeric parity against a checkpoint is covered by the end-to-end
// weights test.

#include "brovisionml/stylegan3.h"

#include <cstdio>
#include <string>

namespace {

int failures = 0;
void check(bool cond, const char* msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; }
}

using brovisionml::stylegan3::Config;
using brovisionml::stylegan3::SynthesisNetwork;

void check_preset(const Config& cfg, int res, const char* tag) {
    SynthesisNetwork net(cfg);
    auto msg = [&](const std::string& s) {
        static std::string buf;
        buf = std::string(tag) + ": " + s;
        return buf.c_str();
    };

    check(net.num_ws() == cfg.num_layers + 2, msg("num_ws = num_layers+2"));
    check(static_cast<int>(net.layers().size()) == cfg.num_layers + 1,
          msg("layer count = num_layers+1"));

    // Fourier input is always low-res (~36) regardless of output resolution,
    // with the full hidden channel count.
    check(net.input().out_h() == 36, msg("input map 36x36"));
    check(net.input().channels() == 1024, msg("input 1024 channels"));

    // First hidden layer: 36x36, 1024 channels; name L0_36_1024.
    check(net.layers().front().out_size() == 36, msg("L0 out_size 36"));
    check(net.layers().front().out_channels() == 1024, msg("L0 1024 channels"));
    check(net.layer_names().front() == "L0_36_1024", msg("L0 name"));

    // ToRGB (last): img_channels at full resolution; name L{N}_{res}_3.
    check(net.layers().back().out_size() == res, msg("ToRGB out_size = res"));
    check(net.layers().back().out_channels() == cfg.img_channels, msg("ToRGB 3 channels"));
    const std::string want = "L" + std::to_string(cfg.num_layers) + "_" +
                             std::to_string(res) + "_3";
    check(net.layer_names().back() == want, msg("ToRGB name"));
}

}  // namespace

int main() {
    check_preset(Config::r256(),  256,  "r256");
    check_preset(Config::r512(),  512,  "r512");
    check_preset(Config::r1024(), 1024, "r1024");

    if (failures == 0) { std::printf("test_stylegan3_synthesis: OK\n"); return 0; }
    std::printf("test_stylegan3_synthesis: %d failure(s)\n", failures);
    return 1;
}
