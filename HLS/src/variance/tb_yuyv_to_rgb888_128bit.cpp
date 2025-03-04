#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdint>
#include "ap_int.h"
#include "hls_stream.h"
#include "yuyv_to_rgb888.h"
#include "image_tile_conversion.hpp"

//------------------------------------------------------------------------------
// AXI stream data type definition (128-bit data word)
struct axis_t {
    ap_uint<128> data;
    bool last;
};

// Helper to clamp a float to [0..255], then cast to uint8_t.
static inline uint8_t clamp_uint8(float x)
{
    if (x < 0.f)   x = 0.f;
    if (x > 255.f) x = 255.f;
    return static_cast<uint8_t>(x);
}

//------------------------------------------------------------------------------
// Reference conversion: YUYV => RGB888 (3 bytes/pixel)
std::vector<uint8_t> yuyv_to_rgb888_convertor(
    const std::vector<uint8_t>& yuyv,
    int width,
    int height)
{
    // Allocate space for (width*height) RGB triples
    std::vector<uint8_t> rgb888(width * height * 3, 0);

    // Process 2 pixels at a time => 4 bytes: [Y0, U, Y1, V]
    for (int i = 0; i < width * height; i += 2) {
        float y0 = static_cast<float>(yuyv[i*2 + 0]);
        float u  = static_cast<float>(yuyv[i*2 + 1]) - 128.f;
        float y1 = static_cast<float>(yuyv[i*2 + 2]);
        float v  = static_cast<float>(yuyv[i*2 + 3]) - 128.f;

        float r0f = y0 + 1.403f * v;
        float g0f = y0 - 0.344f * u - 0.714f * v;
        float b0f = y0 + 1.770f * u;

        float r1f = y1 + 1.403f * v;
        float g1f = y1 - 0.344f * u - 0.714f * v;
        float b1f = y1 + 1.770f * u;

        uint8_t r0 = clamp_uint8(r0f);
        uint8_t g0 = clamp_uint8(g0f);
        uint8_t b0 = clamp_uint8(b0f);

        uint8_t r1 = clamp_uint8(r1f);
        uint8_t g1 = clamp_uint8(g1f);
        uint8_t b1 = clamp_uint8(b1f);

        // Pixel i
        rgb888[i*3 + 0] = r0;
        rgb888[i*3 + 1] = g0;
        rgb888[i*3 + 2] = b0;

        // Pixel i+1
        rgb888[(i+1)*3 + 0] = r1;
        rgb888[(i+1)*3 + 1] = g1;
        rgb888[(i+1)*3 + 2] = b1;
    }
    return rgb888;
}

//------------------------------------------------------------------------------
// Compare expected vs. received outputs
void compare_outputs(
    const std::vector<uint8_t>& expected_output,
    const std::vector<uint8_t>& received_output,
    int& pass_count,
    int& fail_count,
    const std::vector<uint8_t>& /*input_data*/)
{
    if (expected_output == received_output) {
        pass_count++;
        std::cout << "Test PASSED!" << std::endl;
    } else {
        fail_count++;
        std::cout << "Test FAILED!" << std::endl;
        // You could add debug printing here if you like
    }
}

//------------------------------------------------------------------------------
// Send AXI stream input in 128-bit words
void send_axi_stream_input_128bit(
    hls::stream<axis_t> &in_stream,
    const std::vector<uint8_t> &input_data,
    int width,
    int height,
    int in_channels)
{
    int total_bytes = width * height * in_channels;
    for (int i = 0; i < total_bytes; i += 16) {
        ap_uint<128> tdata = 0;
        for (int b = 0; b < 16; b++) {
            int index = i + b;
            uint8_t value = (index < total_bytes) ? input_data[index] : 0;
            tdata.range(8*b + 7, 8*b) = value;
        }
        axis_t val;
        val.data = tdata;
        val.last = ((i + 16) >= total_bytes);
        in_stream.write(val);
    }
}

//------------------------------------------------------------------------------
// Receive AXI stream output in 128-bit words
void receive_axi_stream_output_128bit(
    hls::stream<axis_t> &out_stream,
    std::vector<uint8_t> &out_data,
    int out_width,
    int out_height,
    int out_channels)
{
    int total_out_bytes = out_width * out_height * out_channels;
    out_data.clear();
    out_data.reserve(total_out_bytes);

    int byte_count = 0;
    while (byte_count < total_out_bytes) {
        // Read one 128-bit word
        axis_t val = out_stream.read();
        ap_uint<128> w = val.data;
        // Extract up to 16 bytes
        for (int b = 0; b < 16 && byte_count < total_out_bytes; b++) {
            uint8_t byte_val = static_cast<uint8_t>(w.range(8*b + 7, 8*b));
            out_data.push_back(byte_val);
            byte_count++;
        }
        if (val.last) break;
    }
}

//------------------------------------------------------------------------------
// Main testbench
int main()
{
    int num_tests  = 5;
    int pass_count = 0;
    int fail_count = 0;

    int width       = 28;
    int height      = 28;
    // YUYV = 2 bytes/pixel, so in_channels=2
    // If your DUT produces 3 bytes/pixel (RGB888) without padding, set out_channels=3
    // (If it produces 4 bytes with a padding byte, set out_channels=4
    //  and adapt compare function accordingly.)
    int in_channels  = 2;
    int out_channels = 3;

    //------------------------------------------------------------------------
    // Known "coin tile" self test
    std::cout << "\nRunning coin tile conversion self test case ...\n";

    //   extern const uint8_t conversion_tile_yuyv[];
    //   extern const size_t  conversion_tile_yuyv_len;
    //   extern const uint8_t conversion_tile_rgb[];
    //   extern const size_t  conversion_tile_rgb_len;

    // Convert them into std::vectors
    std::vector<uint8_t> my_conversion_tile_yuyv(
        conversion_tile_yuyv,
        conversion_tile_yuyv + sizeof(conversion_tile_yuyv)/sizeof(conversion_tile_yuyv[0])
    );
    std::vector<uint8_t> my_conversion_tile_rgb(
        conversion_tile_rgb,
        conversion_tile_rgb + sizeof(conversion_tile_rgb)/sizeof(conversion_tile_rgb[0])
    );

    // Use our reference function to convert YUYV => RGB888
    std::vector<uint8_t> rec_output =
        yuyv_to_rgb888_convertor(my_conversion_tile_yuyv, width, height);

    // Compare with the known "golden" RGB for the coin tile
    compare_outputs(my_conversion_tile_rgb, rec_output,
                    pass_count, fail_count, my_conversion_tile_yuyv);

    // Reset pass/fail counts after the coin tile test if you like
    pass_count = 0;
    fail_count = 0;

    //------------------------------------------------------------------------
    // Random stimulus tests
    for (int t = 0; t < num_tests; ++t) {
        std::cout << "\nRunning test case " << (t + 1) << "..." << std::endl;

        // Generate random YUYV input
        std::vector<uint8_t> input_data(width * height * in_channels);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);
        for (auto &val : input_data) {
            val = static_cast<uint8_t>(dis(gen));
        }

        std::vector<uint8_t> expected_output =
            yuyv_to_rgb888_convertor(input_data, width, height);

        // Streams for DUT
        hls::stream<axis_t> in_stream("in_stream");
        hls::stream<axis_t> out_stream("out_stream");

        // Send YUYV input via 128-bit AXI
        send_axi_stream_input_128bit(in_stream, input_data, width, height, in_channels);

        // Call the DUT (commented out if you haven't implemented it):
        // yuyv_converter(in_stream, out_stream);

        // The DUT presumably outputs RGB888 => 3 bytes/pixel
        std::vector<uint8_t> received_output;
        receive_axi_stream_output_128bit(out_stream, received_output,
                                         width, height, out_channels);

        // Compare
        compare_outputs(expected_output, received_output,
                        pass_count, fail_count, input_data);
    }

    // Final summary
    std::cout << "\nTest Summary:\n";
    std::cout << "Total tests run: " << num_tests << "\n";
    std::cout << "Passed: " << pass_count << "\n";
    std::cout << "Failed: " << fail_count << "\n";

    return 0;
}
