#include <iostream>
#include "VarianceAndRGB.h"

int main() {
    hls::stream<ap_uint<32>> pixel_stream;
    hls::stream<ap_uint<32>> output_stream_high;
    hls::stream<ap_uint<32>> output_stream_low;
    float threshold = 50.0f;
    ap_uint<2> override_mode = 0; // 0 is default, 1 is all convolution, 2 is all interpolation

    // Simulate 28x28 pixels in YUYV422 format
    for (int i = 0; i < PIXEL_COUNT / 2; ++i) {
        ap_uint<8> Y0 = i % 256;
        ap_uint<8> U = 128;
        ap_uint<8> Y1 = (i + 1) % 256;
        ap_uint<8> V = 128;
        ap_uint<32> pixel_data = (V << 24) | (Y1 << 16) | (U << 8) | Y0;
        pixel_stream.write(pixel_data);
    }

    process_tile(pixel_stream, output_stream_high, output_stream_low, threshold, override_mode);

    while (!output_stream_high.empty()) {
        std::cout << "Convolution: " << std::hex << output_stream_high.read() << std::endl;
    }
    while (!output_stream_low.empty()) {
        std::cout << "Interpolation: " << std::hex << output_stream_low.read() << std::endl;
    }

    return 0;
}
