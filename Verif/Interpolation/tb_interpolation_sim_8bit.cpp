#include <hls_stream.h>
#include <ap_int.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <random>
#include <cmath>

// Top-level function prototype (as defined in your HLS design)
void Interpolation_v1(
    hls::stream<ap_uint<32>>& image_r_TDATA,
    hls::stream<ap_uint<32>>& featureMap_TDATA,
    ap_uint<8> loadedInfo);

// Packing/unpacking pixels into/from TDATA
// Pixel: R(8 bits), G(8 bits), B(8 bits), total 24 bits
// TDATA (32 bits) layout: [31:24] = 0 padding, [23:16] = R, [15:8] = G, [7:0] = B

uint32_t pack_pixel_to_32(uint8_t R, uint8_t G, uint8_t B) {
    uint32_t tdata = 0;
    tdata |= ((uint32_t)R << 16);
    tdata |= ((uint32_t)G << 8);
    tdata |= (uint32_t)B;
    return tdata;
}

void unpack_pixel_from_32(uint32_t tdata, uint8_t &R, uint8_t &G, uint8_t &B) {
    R = (tdata >> 16) & 0xFF;
    G = (tdata >> 8) & 0xFF;
    B = tdata & 0xFF;
}

// Generate a random image (R,G,B each 8 bits)
std::vector<uint8_t> generate_random_image(int width, int height, int channels = 3) {
    std::vector<uint8_t> image(width * height * channels);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    for (auto& pixel_val : image) {
        pixel_val = dis(gen);
    }
    return image;
}

// Bilinear interpolation function for 8-bit RGB
std::vector<uint8_t> bilinearInterpolation(
    const std::vector<uint8_t>& image, int width, int height, int channels, float scale)
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

            // Get the four neighboring pixels
            // Pixel format: [R,G,B]
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

// Send input data to the design (one pixel per TDATA)
void send_axi_stream_input(hls::stream<ap_uint<32>>& image_r_TDATA, const std::vector<uint8_t>& data, int width, int height, int channels) {
    // Each pixel: R,G,B = 24 bits
    // We'll send each pixel as a single 32-bit word (with top 8 bits zeroed).
    int num_pixels = width * height;
    for (int i = 0; i < num_pixels; i++) {
        uint8_t R = data[i*channels + 0];
        uint8_t G = data[i*channels + 1];
        uint8_t B = data[i*channels + 2];
        uint32_t tdata = pack_pixel_to_32(R, G, B);
        image_r_TDATA.write(tdata);
    }
}

// Receive output data from the design
void receive_axi_stream_output(hls::stream<ap_uint<32>>& featureMap_TDATA, std::vector<uint8_t>& received_data, size_t expected_size, int channels) {
    // expected_size is in terms of total bytes (R,G,B for each pixel).
    int num_pixels = (int)(expected_size / channels);
    for (int i = 0; i < num_pixels; i++) {
        uint32_t tdata = featureMap_TDATA.read();
        uint8_t R, G, B;
        unpack_pixel_from_32(tdata, R, G, B);
        received_data.push_back(R);
        received_data.push_back(G);
        received_data.push_back(B);
    }
}

// Compare outputs
void compare_outputs(const std::vector<uint8_t>& expected_output, const std::vector<uint8_t>& received_output, int& pass_count, int& fail_count, const std::vector<uint8_t>& input_data) {
    bool passed = (expected_output == received_output);
    if (passed) {
        pass_count++;
        std::cout << "Test PASSED!" << std::endl;
    } else {
        fail_count++;
        std::cout << "Test FAILED!" << std::endl;

        // Print input data
        std::cout << "Input Data (R,G,B):" << std::endl;
        for (size_t i = 0; i < input_data.size(); i+=3) {
            std::cout << "(" << (int)input_data[i] << "," << (int)input_data[i+1] << "," << (int)input_data[i+2] << ") ";
        }
        std::cout << std::endl;

        // Print expected output
        std::cout << "Expected Output (R,G,B):" << std::endl;
        for (size_t i = 0; i < expected_output.size(); i+=3) {
            std::cout << "(" << (int)expected_output[i] << "," << (int)expected_output[i+1] << "," << (int)expected_output[i+2] << ") ";
        }
        std::cout << std::endl;

        // Print received output
        std::cout << "Received Output (R,G,B):" << std::endl;
        for (size_t i = 0; i < received_output.size(); i+=3) {
            std::cout << "(" << (int)received_output[i] << "," << (int)received_output[i+1] << "," << (int)received_output[i+2] << ") ";
        }
        std::cout << std::endl;

        // Print differences
        for (size_t i = 0; i < received_output.size(); ++i) {
            if (received_output[i] != expected_output[i]) {
                std::cout << "Mismatch at index " << i << ": expected " << (int)expected_output[i]
                          << ", got " << (int)received_output[i] << std::endl;
            }
        }
    }
}

int main() {
    // Number of test cases
    int num_tests = 10;  // Adjust as needed
    int pass_count = 0;
    int fail_count = 0;

    int width = 2;
    int height = 2;
    int channels = 3;
    float scale = 2.0f;

    for (int test_num = 0; test_num < num_tests; ++test_num) {
        std::cout << "Running test case " << (test_num + 1) << "..." << std::endl;

        // Generate random input image
        std::vector<uint8_t> input_data = generate_random_image(width, height, channels);

        // Compute expected output
        std::vector<uint8_t> expected_output = bilinearInterpolation(input_data, width, height, channels, scale);
        size_t expected_output_size = expected_output.size();

        // Create AXI Stream channels
        hls::stream<ap_uint<32>> image_r_TDATA("image_r_TDATA");
        hls::stream<ap_uint<32>> featureMap_TDATA("featureMap_TDATA");

        // Send input data
        send_axi_stream_input(image_r_TDATA, input_data, width, height, channels);

        // Call the top-level function
        ap_uint<8> loadedInfo = 0;  // Adjust as needed
        Interpolation_v1(image_r_TDATA, featureMap_TDATA, loadedInfo);

        // Receive output data
        std::vector<uint8_t> received_output;
        receive_axi_stream_output(featureMap_TDATA, received_output, expected_output_size, channels);

        // Compare outputs
        compare_outputs(expected_output, received_output, pass_count, fail_count, input_data);
    }

    // Print summary
    std::cout << "Test Summary:" << std::endl;
    std::cout << "Total tests run: " << num_tests << std::endl;
    std::cout << "Passed: " << pass_count << std::endl;
    std::cout << "Failed: " << fail_count << std::endl;

    return 0;
}
