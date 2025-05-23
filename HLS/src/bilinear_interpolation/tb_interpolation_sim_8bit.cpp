#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include "bilinear_interpolation_byte_v2.h"
#include "image_tile_coin.hpp"

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
        std::cout << "Input Data (R,G,B):\n";
        for (size_t i = 0; i < 66; i += 3) {
            std::cout << "(" << (int)input_data[i] << "," << (int)input_data[i+1]
                      << "," << (int)input_data[i+2] << ") ";
        }
        pass_count++;
        std::cout << "\nTest PASSED!" << std::endl;
    } else {
        fail_count++;
        std::cout << "Test FAILED!" << std::endl;
        std::cout << "Input Data (R,G,B):\n";
        for (size_t i = 0; i < input_data.size(); i += 3) {
            std::cout << "(" << (int)input_data[i] << "," << (int)input_data[i+1]
                      << "," << (int)input_data[i+2] << ") ";
        }
        std::cout << "\nExpected:\n";
        for (size_t i = 0; i < expected_output.size(); i += 3) {
            std::cout << "(" << (int)expected_output[i] << "," << (int)expected_output[i+1]
                      << "," << (int)expected_output[i+2] << ") ";
        }
        std::cout << "\nReceived:\n";
        for (size_t i = 0; i < received_output.size(); i += 3) {
            std::cout << "(" << (int)received_output[i] << "," << (int)received_output[i+1]
                      << "," << (int)received_output[i+2] << ") ";
        }
        std::cout << std::endl;

        // Print detailed mismatch info for debugging
        for (size_t i = 0; i < received_output.size(); ++i) {
            if (received_output[i] != expected_output[i]) {
                std::cout << "Mismatch at index " << i << ": expected "
                          << (int)expected_output[i]
                          << ", got " << (int)received_output[i] << std::endl;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Send AXI stream input (8 bits)
// -----------------------------------------------------------------------------
void send_axi_stream_input_8bit(hls::stream<axis_t> &in_stream,
                                const std::vector<uint8_t> &input_data,
                                int width, int height, int channels)
{
    int total_bytes = width * height * channels;

    for (int i = 0; i < total_bytes; i++) {

        // Wait for TREADY => in_stream not full
        while (in_stream.full()) {
            // Wait
        }

        axis_t val;
        val.data = input_data[i];
        val.last = (i == (total_bytes - 1)); // last byte in the image?

        // // Optional debug
        // std::cout << "Sending byte " << i 
        //           << " = 0x" << std::hex << (int)input_data[i]
        //           << " (last=" << val.last << ")" 
        //           << std::dec << std::endl;

        in_stream.write(val);  // TVALID=1, TREADY=1 => transfer
    }
}

// -----------------------------------------------------------------------------
// Receive AXI stream output (8 bits)
// -----------------------------------------------------------------------------
void receive_axi_stream_output_8bit(hls::stream<axis_t> &out_stream,
                                    std::vector<uint8_t> &out_data,
                                    int out_width, int out_height, int out_channels)
{
    int total_out_bytes = out_width * out_height * out_channels;

    out_data.clear();
    out_data.reserve(total_out_bytes);

    int byte_count = 0;

    while (byte_count < total_out_bytes) {
        // Wait for TVALID => out_stream not empty
        while (out_stream.empty()) {
            // Wait
        }

        axis_t val = out_stream.read();
        uint8_t b  = val.data;

        out_data.push_back(b);
        byte_count++;

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
    int num_tests  = 5;
    int pass_count = 0;
    int fail_count = 0;

    int width    = 28;
    int height   = 28;
    int channels = 3;
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

        // Generate random stimulus data for one 28x28x3 image
        int total_pixels = width * height;
        std::vector<uint8_t> input_data(total_pixels * channels);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);
        for (auto &val : input_data) {
            val = dis(gen);
        }

        // CPU reference
        std::vector<uint8_t> expected_output =
            bilinearInterpolation(input_data, width, height, channels, scale);

        // HLS axi streams
        hls::stream<axis_t> in_stream("in_stream");
        hls::stream<axis_t> out_stream("out_stream");

        send_axi_stream_input_8bit(in_stream, input_data, width, height, channels);

        // DUT
        bilinear_interpolation_byte(in_stream, out_stream);

        int out_w = (int)(width * scale);
        int out_h = (int)(height * scale);
        std::vector<uint8_t> received_output;
        receive_axi_stream_output_8bit(out_stream, received_output, out_w, out_h, channels);

        compare_outputs(expected_output, received_output,
                        pass_count, fail_count, input_data);
    }

    std::cout << "\nTest Summary:\n";
    std::cout << "Total tests run: " << num_tests << "\n";
    std::cout << "Passed: " << pass_count << "\n";
    std::cout << "Failed: " << fail_count << "\n";

    return 0;
}
