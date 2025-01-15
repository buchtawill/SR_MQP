#include "bilinear_interpolation.h"
#include <iostream>
#include <hls_stream.h>
#include <ap_int.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

typedef struct {
    ap_uint<32> data;
    bool last;
} axis32_t;

// CPU-based bilinear interpolation
std::vector<uint8_t> bilinearInterpolation(
    const std::vector<uint8_t>& image,
    int width,
    int height,
    int channels,
    float scale)
{
    int newWidth = static_cast<int>(width * scale);
    int newHeight = static_cast<int>(height * scale);
    std::vector<uint8_t> outputImage(newWidth * newHeight * channels);

    for (int y_out = 0; y_out < newHeight; ++y_out) {
        for (int x_out = 0; x_out < newWidth; ++x_out) {
            float x_in = x_out / scale;
            float y_in = y_out / scale;
            int x0 = static_cast<int>(std::floor(x_in));
            int y0 = static_cast<int>(std::floor(y_in));
            int x1 = std::min(x0 + 1, width - 1);
            int y1 = std::min(y0 + 1, height - 1);
            float dx = x_in - x0;
            float dy = y_in - y0;
            float w00 = (1 - dx)*(1 - dy);
            float w10 = dx*(1 - dy);
            float w01 = (1 - dx)*dy;
            float w11 = dx*dy;

            auto get_pixel = [&](int x, int y, int c) {
                return (float)image[(y * width + x)*channels + c];
            };

            float R00 = get_pixel(x0, y0, 0);
            float G00 = get_pixel(x0, y0, 1);
            float B00 = get_pixel(x0, y0, 2);

            float R10 = get_pixel(x1, y0, 0);
            float G10 = get_pixel(x1, y0, 1);
            float B10 = get_pixel(x1, y0, 2);

            float R01 = get_pixel(x0, y1, 0);
            float G01 = get_pixel(x0, y1, 1);
            float B01 = get_pixel(x0, y1, 2);

            float R11 = get_pixel(x1, y1, 0);
            float G11 = get_pixel(x1, y1, 1);
            float B11 = get_pixel(x1, y1, 2);

            float R_val = w00*R00 + w10*R10 + w01*R01 + w11*R11;
            float G_val = w00*G00 + w10*G10 + w01*G01 + w11*G11;
            float B_val = w00*B00 + w10*B10 + w01*B01 + w11*B11;

            int idx = (y_out * newWidth + x_out)*channels;
            outputImage[idx+0] = (uint8_t)std::round(R_val);
            outputImage[idx+1] = (uint8_t)std::round(G_val);
            outputImage[idx+2] = (uint8_t)std::round(B_val);
        }
    }
    return outputImage;
}

// Compare function
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
        fail_count++;
        std::cout << "Test FAILED!" << std::endl;

        std::cout << "Input Data (R,G,B):" << std::endl;
        for (size_t i = 0; i < input_data.size(); i += 3) {
            std::cout << "(" << (int)input_data[i] << "," << (int)input_data[i+1]
                      << "," << (int)input_data[i+2] << ") ";
        }
        std::cout << std::endl;

        std::cout << "Expected Output (R,G,B):" << std::endl;
        for (size_t i = 0; i < expected_output.size(); i += 3) {
            std::cout << "(" << (int)expected_output[i] << "," << (int)expected_output[i+1]
                      << "," << (int)expected_output[i+2] << ") ";
        }
        std::cout << std::endl;

        std::cout << "Received Output (R,G,B):" << std::endl;
        for (size_t i = 0; i < received_output.size(); i += 3) {
            std::cout << "(" << (int)received_output[i] << "," << (int)received_output[i+1]
                      << "," << (int)received_output[i+2] << ") ";
        }
        std::cout << std::endl;

        for (size_t i = 0; i < received_output.size(); ++i) {
            if (received_output[i] != expected_output[i]) {
                std::cout << "Mismatch at index " << i << ": expected "
                          << (int)expected_output[i]
                          << ", got " << (int)received_output[i] << std::endl;
            }
        }
    }
}

// Rename the hardware function signature to use axis32_t
extern void bilinear_interpolation(hls::stream<axis32_t> &in_stream,
                                   hls::stream<axis32_t> &out_stream);

// Send data
void send_axi_stream_input(hls::stream<axis32_t> &in_stream,
                           const std::vector<uint8_t> &input_data,
                           int width, int height, int channels)
{
    int total_pixels = width * height;
    for (int px = 0; px < total_pixels; px++) {
        uint8_t R0 = input_data[px*3 + 0];
        uint8_t G0 = input_data[px*3 + 1];
        uint8_t B0 = input_data[px*3 + 2];
        uint8_t nextR = 0;
        if ((px + 1) < total_pixels) {
            nextR = input_data[(px+1)*3 + 0];
        }

        ap_uint<32> tdata = 0;
        tdata |= (ap_uint<32>)B0;
        tdata |= ((ap_uint<32>)G0 << 8);
        tdata |= ((ap_uint<32>)R0 << 16);
        tdata |= ((ap_uint<32>)nextR << 24);

        axis32_t val;
        val.data = tdata;
        val.last = (px == (total_pixels - 1));
        in_stream.write(val);
    }
}

// Receive data
void receive_axi_stream_output(hls::stream<axis32_t> &out_stream,
                               std::vector<uint8_t> &out_data,
                               int out_width, int out_height, int out_channels)
{
    int out_pixels = out_width * out_height;
    out_data.clear();
    out_data.reserve(out_pixels * out_channels);

    int px_count = 0;
    while (px_count < out_pixels) {
        axis32_t val = out_stream.read();
        ap_uint<32> w = val.data;

        uint8_t B0    = (uint8_t)((w >>  0) & 0xFF);
        uint8_t G0    = (uint8_t)((w >>  8) & 0xFF);
        uint8_t R0    = (uint8_t)((w >> 16) & 0xFF);
        uint8_t nextR = (uint8_t)((w >> 24) & 0xFF);

        out_data.push_back(R0);
        out_data.push_back(G0);
        out_data.push_back(B0);
        px_count++;

        if (px_count < out_pixels) {
            axis32_t val2 = out_stream.read();
            ap_uint<32> w2 = val2.data;

            uint8_t B1     = (uint8_t)((w2 >>  0) & 0xFF);
            uint8_t G1     = (uint8_t)((w2 >>  8) & 0xFF);
            uint8_t R1     = (uint8_t)((w2 >> 16) & 0xFF);
            uint8_t nextR2 = (uint8_t)((w2 >> 24) & 0xFF);

            out_data.push_back(nextR);
            out_data.push_back(G1);
            out_data.push_back(B1);
            px_count++;

            if (px_count < out_pixels) {
                out_data.push_back(R1);
                out_data.push_back(nextR2);
                out_data.push_back(0);
                px_count++;
            }
            if (val2.last && (px_count < out_pixels)) {
                break;
            }
        }
        if (val.last) {
            break;
        }
    }
    if ((int)out_data.size() > (out_pixels * out_channels)) {
        out_data.resize(out_pixels * out_channels);
    }
}

int main()
{
    int num_tests  = 5;
    int pass_count = 0;
    int fail_count = 0;

    int width    = 4;
    int height   = 4;
    int channels = 3;
    float scale  = 2.0f;

    for (int t = 0; t < num_tests; ++t) {
        std::cout << "\nRunning test case " << (t+1) << "...\n";
        int total_pixels = width * height;
        std::vector<uint8_t> input_data(total_pixels * channels);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);
        for (auto &val : input_data) val = dis(gen);

        std::vector<uint8_t> expected_output =
            bilinearInterpolation(input_data, width, height, channels, scale);

        hls::stream<axis32_t> in_stream("in_stream"), out_stream("out_stream");
        send_axi_stream_input(in_stream, input_data, width, height, channels);
        bilinear_interpolation(in_stream, out_stream);

        int out_w = (int)(width * scale);
        int out_h = (int)(height * scale);
        std::vector<uint8_t> received_output;
        receive_axi_stream_output(out_stream, received_output, out_w, out_h, channels);

        compare_outputs(expected_output, received_output, pass_count, fail_count, input_data);
    }

    std::cout << "\nTest Summary:\n";
    std::cout << "Total tests run: " << num_tests << "\n";
    std::cout << "Passed: " << pass_count << "\n";
    std::cout << "Failed: " << fail_count << "\n";
    return 0;
}
