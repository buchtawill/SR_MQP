#include <iostream>
#include "VarianceAndRGB.h"

int main() {
    hls::stream<axis_t> pixel_stream_in; // redefine streams
    hls::stream<axis_t> conv_out;
    hls::stream<axis_t> interp_out;
    unsigned int threshold = 1000;
    ap_uint<2> override_mode = 0; // 0 is default, 1 is all convolution, 2 is all interpolation

    axis_t pixel_data; // 128-bit wide transfer (4 YUYV pixel packages --> 8 pixels)

    pixel_data.data = 0;
    pixel_data.last = 0;
    // Simulate 28x28 pixels in YUYV422 format --> HIGH VARIANCE
    for (int i = 0; i < NUM_TRANSFERS; i++) {
        for (int j = 0; j < 4; j++) { // 4 32-bit YUYV pixel groups per transfer
        	int Y0 = (i*4 + j) % 256;
			int U = 128;
			int Y1 = (i*4 + j + 1) % 256;
			int V = 128;
			YUYV32 tmp_YUYV = (V << 24) | (Y1 << 16) | (U << 8) | Y0;
			pixel_data.data.range((j+1)*32-1, j*32) = tmp_YUYV;
        }

//        std::cout << "INFO [testbench] Package transfer: " << std::hex << pixel_data.data << std::endl;
        pixel_data.last = (i == NUM_TRANSFERS - 1);
        pixel_stream_in.write(pixel_data);
    }

    process_tile(pixel_stream_in, conv_out, interp_out, threshold, override_mode);

    while (!conv_out.empty()) {
    	// should result in variance = 4904
        std::cout << "Convolution: " << std::hex << conv_out.read().data << std::endl;
    }
    while (!interp_out.empty()) {
        std::cout << "Interpolation: " << std::hex << interp_out.read().data << std::endl;
    }

    // Simulate 28x28 pixels in YUYV422 format --> LOW VARIANCE
    for (int i = 0; i < NUM_TRANSFERS; i++) {
		for (int j = 0; j < 4; j++) {
    		int Y0 = (i*4 + j) % 10;
			int U = 128;
			int Y1 = (i*4 + j + 1) % 10;
			int V = 128;
			YUYV32 tmp_YUYV = (V << 24) | (Y1 << 16) | (U << 8) | Y0;
			pixel_data.data.range((j+1)*32-1, j*32) = tmp_YUYV;
		}
        pixel_stream_in.write(pixel_data);
    }

    process_tile(pixel_stream_in, conv_out, interp_out, threshold, override_mode);

    while (!conv_out.empty()) {
        std::cout << "Convolution: " << std::hex << conv_out.read().data << std::endl;
    }
    while (!interp_out.empty()) {
    	// should result in variance = 8
        std::cout << "Interpolation: " << std::hex << interp_out.read().data << std::endl;
    }

    return 0;
}
