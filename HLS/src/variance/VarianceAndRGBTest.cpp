#include <iostream>
#include "VarianceAndRGB.h"
#include "../image_tile_conversion.hpp"
#include <cmath>

int test_rgb_convert();

int main() {
    hls::stream<axis_t> pixel_stream_in;
    hls::stream<axis_t> conv_out;
    hls::stream<axis_t> interp_out;
    unsigned int threshold = 1000;
    ap_uint<2> override_mode = 0; // 0 is default, 1 is all convolution, 2 is all interpolation

    axis_t pixel_data; // 128-bit wide transfer (4 YUYV pixel packages --> 8 pixels)

    pixel_data.data = 0;
    pixel_data.keep = 0xffff;
    pixel_data.strb = 0xffff;
    pixel_data.last = 0;

    // Simulate 32x32 pixels in YUYV422 format --> HIGH VARIANCE
    for (int i = 0; i < YUYV_NUM_TRANSFERS; i++) {
        for (int j = 0; j < 4; j++) { // 4 32-bit YUYV pixel groups per transfer
        	int Y0 = (i*4 + j) % 256;
			int U = 128;
			int Y1 = (i*4 + j + 1) % 256;
			int V = 128;
			ap_32 tmp_YUYV = (V << 24) | (Y1 << 16) | (U << 8) | Y0;
			pixel_data.data.range((j+1)*32-1, j*32) = tmp_YUYV;
        }

//        std::cout << "INFO [testbench] Package transfer: " << std::hex << pixel_data.data << std::endl;
        pixel_data.last = (i == YUYV_NUM_TRANSFERS - 1);
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

    // Simulate 32x32 pixels in YUYV422 format --> LOW VARIANCE
    for (int i = 0; i < YUYV_NUM_TRANSFERS; i++) {
		for (int j = 0; j < 4; j++) {
    		int Y0 = (i*4 + j) % 10;
			int U = 128;
			int Y1 = (i*4 + j + 1) % 10;
			int V = 128;
			ap_32 tmp_YUYV = (V << 24) | (Y1 << 16) | (U << 8) | Y0;
			pixel_data.data.range((j+1)*32-1, j*32) = tmp_YUYV;
		}
        pixel_data.last = (i == YUYV_NUM_TRANSFERS - 1);
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


    // Send image_tile_conversion.hpp data through the whole block
	for (int i = 0; i < YUYV_NUM_TRANSFERS; i++){
	    for (int j = 0; j < 16; j++){
	        pixel_data.data |= ((ap_uint_128)conversion_tile_yuyv[i * 16 + j]) << (8 * j);
	    }
        pixel_data.last = (i == YUYV_NUM_TRANSFERS - 1);
        pixel_stream_in.write(pixel_data);
	}

    process_tile(pixel_stream_in, conv_out, interp_out, threshold, override_mode);

    while (!conv_out.empty()) {
        std::cout << "Convolution: " << std::hex << conv_out.read().data << std::endl;
    }
    while (!interp_out.empty()) {
        std::cout << "Interpolation: " << std::hex << interp_out.read().data << std::endl;
    }

    test_rgb_convert();

    return 0;
}


int test_rgb_convert(){
	ap_uint_128 function_output[RGB_NUM_TRANSFERS];
	ap_uint_128 packed_YUYV_input[YUYV_NUM_TRANSFERS];
	ap_uint_128 packed_data = 0;

	for (int i = 0; i < YUYV_NUM_TRANSFERS; i++){
	    ap_uint_128 packed_data = 0;
	    for (int j = 0; j < 16; j++){
	        packed_data |= ((ap_uint_128)conversion_tile_yuyv[i * 16 + j]) << (8 * j);
	    }
	    packed_YUYV_input[i] = packed_data;
	}


	rgb_convert(function_output, packed_YUYV_input);

	    ap_8 extracted_rgb[RGB_NUM_TRANSFERS];
	    int index = 0;

	    for (int i = 0; i < RGB_NUM_TRANSFERS; i++) {
			for (int j = 0; j < 4; j++) {
				ap_32 pixel = function_output[i].range((j * 32) + 31, j * 32);
				extracted_rgb[index++] = (pixel >> 0) & 0xFF;  // R
				extracted_rgb[index++] = (pixel >> 8) & 0xFF;  // G
				extracted_rgb[index++] = (pixel >> 16) & 0xFF; // B
			}
		}

	    // compare extracted and expected values
		bool all_pass = true;
		bool this_pass;
		int biggest_diff = 0;
		int diff = 0;
		int passes = 0;
		int fails = 0;
		int total_tests = 0;
		std::cout << "Expected vs. Output RGB Values:\n";
		for (int i = 0; i < PIXEL_COUNT; i++) {
			diff = abs(conversion_tile_rgb[i] - extracted_rgb[i]);
			if (diff > ACCEPTABLE_ERROR) {
				if (diff > biggest_diff) biggest_diff = diff;

				std::cout << "Expected: " << std::setw(3) << (int)conversion_tile_rgb[i] << " | Output: " << std::setw(3) << (int)extracted_rgb[i] << " |   FAIL 		DIFFERENCE = " << diff << "\n" ;
				all_pass = false;
				this_pass = false;
				fails++;
				total_tests++;
			}
			else {
				std::cout << "Expected: " << std::setw(3) << (int)conversion_tile_rgb[i] << " | Output: " << std::setw(3) << (int)extracted_rgb[i] << " |   PASS 		DIFFERENCE = " << diff << "\n" ;
				this_pass = true;
				passes++;
				total_tests++;
			}
		}

		if (all_pass) {
			std::cout << "All Tests Passed!\n";
		} else {
			std::cout << "Test Failed!\n";
			std::cout << "Biggest Pixel Difference: " << biggest_diff << "\n" ;
		}

		std::cout << "PASSES: " << passes << " OUT OF " << total_tests << "\n";
		std::cout << "FAILS: " << fails << " OUT OF " << total_tests << "\n";

	return 0;
}
