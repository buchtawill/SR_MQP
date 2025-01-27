#include <iostream>
#include "VarianceAndRGB.h"

int main() {
    hls::stream<axis_t> pixel_stream_in; // redefine streams
    hls::stream<axis_t> conv_out;
    hls::stream<axis_t> interp_out;
    unsigned int threshold = 50;
    ap_uint<2> override_mode = 0; // 0 is default, 1 is all convolution, 2 is all interpolation

    axis_t pixel_data;

    // Simulate 28x28 pixels in YUYV422 format
    for (int i = 0; i < PIXEL_COUNT / 2; i++) {
        int Y0 = i % 256;
        int U = 128;
        int Y1 = (i + 1) % 256;
        int V = 128;
        pixel_data.data = (V << 24) | (Y1 << 16) | (U << 8) | Y0;
        pixel_stream_in.write(pixel_data);
    }

    // send TLAST and TKEEP - check Langa's code
    // keep TREADY high until stream is ended - depends on how data is read

    process_tile(pixel_stream_in, conv_out, interp_out, threshold, override_mode);

    while (!conv_out.empty()) {
        std::cout << "Convolution: " << std::hex << conv_out.read().data << std::endl;
    }
    while (!interp_out.empty()) {
        std::cout << "Interpolation: " << std::hex << interp_out.read().data << std::endl;
    }

    return 0;
}
