#include <iostream>
#include "VarianceAndRGB.h"

int main() {
    hls::stream<data_stream> pixel_stream_in;
    hls::stream<data_stream> conv_out;
    hls::stream<data_stream> interp_out;
    float threshold = 50.0f;
    ap_uint<2> override_mode = 0; // 0 is default, 1 is all convolution, 2 is all interpolation

    // Simulate 28x28 pixels in YUYV422 format
    for (int i = 0; i < PIXEL_COUNT / 2; i++) {
        pixel_component Y0 = i % 256;
        pixel_component U = 128;
        pixel_component Y1 = (i + 1) % 256;
        pixel_component V = 128;
        data_stream pixel_data = (V << 24) | (Y1 << 16) | (U << 8) | Y0;
        pixel_stream_in.write(pixel_data);
    }

    process_tile(pixel_stream_in, conv_out, interp_out, threshold, override_mode);

    while (!conv_out.empty()) {
        std::cout << "Convolution: " << std::hex << conv_out.read() << std::endl;
    }
    while (!interp_out.empty()) {
        std::cout << "Interpolation: " << std::hex << interp_out.read() << std::endl;
    }

    return 0;
}
