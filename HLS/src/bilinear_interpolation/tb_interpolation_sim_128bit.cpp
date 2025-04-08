#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include "bilinear_interpolation.h"
#include "image_coin_tile.h"

// -----------------------------------------------------------------------------
// Expected interpolation output  
// -----------------------------------------------------------------------------
std::vector<uint8_t> bilinearInterpolation(
    const std::vector<uint8_t>& image,
    int width,
    int height,
    int channels,
    float scale)
{
    int newWidth  = static_cast<int>(width  * scale);
    int newHeight = static_cast<int>(height * scale);

    std::vector<uint8_t> outputImage(newWidth * newHeight * channels);

    float widthRatio  = (newWidth  > 1) ? float(width  - 1) / float(newWidth  - 1) : 0.f;
    float heightRatio = (newHeight > 1) ? float(height - 1) / float(newHeight - 1) : 0.f;

    for (int y_out = 0; y_out < newHeight; ++y_out) {
        for (int x_out = 0; x_out < newWidth; ++x_out) {
            float x_in = x_out * widthRatio;
            float y_in = y_out * heightRatio;

            int x0 = static_cast<int>(std::floor(x_in));
            int y0 = static_cast<int>(std::floor(y_in));
            int x1 = std::min(x0 + 1, width  - 1);
            int y1 = std::min(y0 + 1, height - 1);

            float dx = x_in - x0;
            float dy = y_in - y0;

            float w00 = (1 - dx) * (1 - dy);
            float w10 = dx        * (1 - dy);
            float w01 = (1 - dx) * dy;
            float w11 = dx       * dy;

            auto get_pixel = [&](int x, int y, int c) {
                return static_cast<float>( image[(y * width + x) * channels + c] );
            };

            float R_val = w00 * get_pixel(x0, y0, 0)
                        + w10 * get_pixel(x1, y0, 0)
                        + w01 * get_pixel(x0, y1, 0)
                        + w11 * get_pixel(x1, y1, 0);

            float G_val = w00 * get_pixel(x0, y0, 1)
                        + w10 * get_pixel(x1, y0, 1)
                        + w01 * get_pixel(x0, y1, 1)
                        + w11 * get_pixel(x1, y1, 1);

            float B_val = w00 * get_pixel(x0, y0, 2)
                        + w10 * get_pixel(x1, y0, 2)
                        + w01 * get_pixel(x0, y1, 2)
                        + w11 * get_pixel(x1, y1, 2);

            int idx = (y_out * newWidth + x_out) * channels;
            outputImage[idx + 0] = static_cast<uint8_t>(std::round(R_val));
            outputImage[idx + 1] = static_cast<uint8_t>(std::round(G_val));
            outputImage[idx + 2] = static_cast<uint8_t>(std::round(B_val));
        }
    }
    return outputImage;
}

// -----------------------------------------------------------------------------
// Monitor: Compare Expected vs. DUT output
// -----------------------------------------------------------------------------
void compare_outputs(const std::vector<uint8_t>& expected_output,
                     const std::vector<uint8_t>& received_output,
                     int& pass_count,
                     int& fail_count,
                     const std::vector<uint8_t>& input_data)
{
    bool passed = (expected_output == received_output);
    if (passed) {
        pass_count++;
        std::cout << "Test PASSED!" << std::endl;
    } else {
        std::cout << "Input Data (R,G,B):\n";
        for (size_t i = 0; i < input_data.size(); i += MARGIN_OF_ERROR) {
            std::cout << "(" << (int)input_data[i] << "," << (int)input_data[i+1]
                      << "," << (int)input_data[i+2] << ") ";
        }
        std::cout << "\nExpected:\n";
        for (size_t i = 0; i < expected_output.size(); i += MARGIN_OF_ERROR) {
            std::cout << "(" << (int)expected_output[i] << "," << (int)expected_output[i+1]
                      << "," << (int)expected_output[i+2] << ") ";
        }
        std::cout << "\nReceived:\n";
        for (size_t i = 0; i < received_output.size(); i += MARGIN_OF_ERROR) {
            std::cout << "(" << (int)received_output[i] << "," << (int)received_output[i+1]
                      << "," << (int)received_output[i+2] << ") ";
        }
        std::cout << std::endl;
        passed = true;

        // Print detailed mismatch info
        for (size_t i = 0; i < received_output.size(); ++i) {
            if (received_output[i] != expected_output[i] && (received_output[i]-expected_output[i]>MARGIN_OF_ERROR ||received_output[i]-expected_output[i]<-MARGIN_OF_ERROR)) {
                if(passed){
                	passed = false;
                	fail_count++;
                    std::cout << "Test FAILED!" << std::endl;
                }
            	std::cout << "Mismatch at index " << i << ": expected "
                          << (int)expected_output[i]
                          << ", got " << (int)received_output[i] << std::endl;
            }
        }
        if(passed){
            pass_count++;
            std::cout << "Test PASSED!" << std::endl;
        }
    }
}

// -----------------------------------------------------------------------------
// Send AXI stream input (128 bits, 16 bytes)
// -----------------------------------------------------------------------------
void send_axi_stream_input_128bit(hls::stream<axis_t> &in_stream,
                                  const std::vector<uint8_t> &input_data,
                                  int width, int height, int channels)
{
    int total_bytes = width * height * (channels-1);
    int padding = 0;

    // Step through the whole tile in chunks of 16 bytes
    for (int i = 0; i < total_bytes; i += 12) {
        // Wait if stream is full (simulate TREADY=0)
        while (in_stream.full()) {
            // Wait
        }

        ap_uint<128> tdata = 0;
        int index = i-1;

        for (int b = 0; b < 16; b++) {
			uint8_t value = 0;
        	if(padding==3){
        		value = 0;
        		padding = 0;
        	}else{
        		padding++;
        		index++;
//        		std::cout << "Index: " << index << "...\n";
				if (index < total_bytes) {
					value = input_data[index];
				}
        	}
			// Place this byte at [8*b + 7 : 8*b] in the 128-bit data
			tdata.range(8*b + 7, 8*b) = value;
        }
        axis_t val;
        val.data = tdata;
        val.last = ((i + 16) >= total_bytes); // last byte in the image?

        // // Optional debug
        // std::cout << "Sending bytes " << i << "-" << (i+15)
        //           << " => 0x" << std::hex << tdata
        //           << " (last=" << val.last << ")" << std::dec << std::endl;

        in_stream.write(val);  // 128-bit transfer
    }
}

// -----------------------------------------------------------------------------
// Receive AXI stream output (128 bits = 16 bytes)
// -----------------------------------------------------------------------------
void receive_axi_stream_output_128bit(hls::stream<axis_t> &out_stream,
                                      std::vector<uint8_t> &out_data,
                                      int out_width, int out_height, int out_channels)
{
    int total_out_bytes = out_width * out_height * out_channels;

    out_data.clear();
    out_data.reserve(total_out_bytes);

    int byte_count = 0;
    int padding = 0;

    while (byte_count < total_out_bytes) {
        // Wait if stream is empty (simulate TVALID=0)
        while (out_stream.empty()) {
            // Wait
        }

        axis_t val = out_stream.read();
        ap_uint<128> w = val.data;

        for (int b = 0; b < 16 && byte_count < total_out_bytes; b++) {
        	if(padding==3){
        		padding = 0;
        		continue;
        	}
        	padding++;
            uint8_t byte_val = (uint8_t)(w.range(8*b + 7, 8*b));
            out_data.push_back(byte_val);
            byte_count++;
        }

        if (val.last) {
            break;
        }
    }
}

// -----------------------------------------------------------------------------
// Main testbench
// -----------------------------------------------------------------------------
int main()
{
    int num_tests  = 10;
    int pass_count = 0;
    int fail_count = 0;

    int width    = 28;
    int height   = 28;
    int channels = 4;
    float scale  = 2.0f;

    // Tile coin self test
    std::cout << "\nRunning tile coin self test case ...\n";

    std::vector<uint8_t> my_coin_tile_low_res;
    std::vector<uint8_t> my_coin_tile_interpolated;
    
    my_coin_tile_low_res.assign(coin_tile_low_res, coin_tile_low_res + sizeof(coin_tile_low_res) / sizeof(coin_tile_low_res[0]));
    my_coin_tile_interpolated.assign(coin_tile_interpolated, coin_tile_interpolated + sizeof(coin_tile_interpolated) / sizeof(coin_tile_interpolated[0]));

    std::vector<uint8_t> rec_output = bilinearInterpolation(my_coin_tile_low_res, width, height, 3, scale);
    compare_outputs(my_coin_tile_interpolated, rec_output, pass_count, fail_count, my_coin_tile_low_res);

    pass_count = 0;
    fail_count = 0;

    for (int t = 0; t < num_tests; ++t) {
        std::cout << "\nRunning test case " << (t+1) << "...\n";

        // Generate random input data for one 28x28x3 image
        int total_pixels = width * height;
        std::vector<uint8_t> input_data(total_pixels * channels);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);
        for (auto &val : input_data) {
            val = dis(gen);
        }

        std::vector<uint8_t> expected_output =
            bilinearInterpolation(input_data, width, height, 3, scale);

        // HLS streams
        hls::stream<axis_t> in_stream("in_stream");
        hls::stream<axis_t> out_stream("out_stream");

        send_axi_stream_input_128bit(in_stream, input_data, width, height, channels);

        // Run DUT
        bilinear_interpolation(in_stream, out_stream);

        int out_w = (int)(width * scale);
        int out_h = (int)(height * scale);
        std::vector<uint8_t> received_output;
        receive_axi_stream_output_128bit(out_stream, received_output, out_w, out_h, channels);

        compare_outputs(expected_output, received_output,
                        pass_count, fail_count, input_data);
    }

    std::cout << "\nTest Summary:\n";
    std::cout << "Total tests run: " << num_tests << "\n";
    std::cout << "Passed: " << pass_count << "\n";
    std::cout << "Failed: " << fail_count << "\n";

    return 0;
}
