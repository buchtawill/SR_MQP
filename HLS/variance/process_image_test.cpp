#include <iostream>
#include <hls_stream.h>
#include "process_image.h"

// Define test dimensions
const int TEST_IMAGE_WIDTH = 56; // Two 28x28 tiles horizontally
const int TEST_IMAGE_HEIGHT = 56; // Two 28x28 tiles vertically

void test_process_image() {
    // Input and output streams
    hls::stream<axi_stream_in> input_stream;
    hls::stream<axi_stream_out> output_stream;

    // Generate synthetic YUYV422 data
    for (int i = 0; i < TEST_IMAGE_HEIGHT; ++i) {
        for (int j = 0; j < TEST_IMAGE_WIDTH; j += 2) {
            axi_stream_in data_in;
            ap_uint<32> yuyv_pixel;

            // YUYV: Y0, U, Y1, V
            ap_uint<8> y0 = (i + j) % 256;
            ap_uint<8> u = 128;
            ap_uint<8> y1 = (i + j + 1) % 256;
            ap_uint<8> v = 128;

            yuyv_pixel.range(7, 0) = y0;
            yuyv_pixel.range(15, 8) = u;
            yuyv_pixel.range(23, 16) = y1;
            yuyv_pixel.range(31, 24) = v;

            data_in.data = yuyv_pixel;
            input_stream.write(data_in);
        }
    }

    // Process image
    process_image(input_stream, output_stream);

    // Read and verify RGB888 output
    while (!output_stream.empty()) {
        axi_stream_out data_out = output_stream.read();
        ap_uint<24> rgb = data_out.data;

        // Extract RGB values
        ap_uint<8> r = rgb.range(23, 16);
        ap_uint<8> g = rgb.range(15, 8);
        ap_uint<8> b = rgb.range(7, 0);

        std::cout << "R: " << (int)r << ", G: " << (int)g << ", B: " << (int)b << std::endl;
    }
}

int main() {
    test_process_image();
    return 0;
}
