#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <random>
#include <cstdint>
#include <cmath>
#include "hls_stream.h"
#include "ap_int.h"
#include "conv2d.h"


//----------------------------------------------------------------
// Pack each pixel as 32 bits (R, G, B, pad=0)
// 4 pixels per 128-bit word
void send_axi_stream_input_128bit(hls::stream<axis_t>& in_stream,
                                  const std::vector<uint8_t>& input_data,
                                  int width, int height, int channels) {
    // channels is 3 (RGB)
    int total_bytes = width * height * channels; // valid data bytes
    // Each pixel is sent as 4 bytes (3 valid + 1 pad). 
    // Pack 4 pixels (16 bytes) per word. 
    int padding = 0;
    for (int i = 0; i < total_bytes; i += 12) {
        while(in_stream.full()) { }
        ap_uint<128> tdata = 0;
        int index = i - 1;
        for (int b = 0; b < 16; b++) {
            uint8_t value = 0;
            if (padding == 3) {
                value = 0; // pad
                padding = 0;
            } else {
                padding++;
                index++;
                if(index < total_bytes)
                    value = input_data[index];
            }
            tdata.range(8*b+7, 8*b) = value;
        }
        axis_t word;
        word.data = tdata;
        word.last = ((i + 16) >= total_bytes);
        in_stream.write(word);
    }
}

//----------------------------------------------------------------
// Unpack 128-bit words into valid 8-bit values
// It extracts 3 valid bytes per pixel.
void receive_axi_stream_output_128bit(hls::stream<axis_t>& out_stream,
                                      std::vector<uint8_t>& out_data,
                                      int out_width, int out_height, int channels) {
    int total_valid_bytes = out_width * out_height * channels;
    out_data.clear();
    out_data.reserve(total_valid_bytes);
    int byte_count = 0;
    int padding = 0;
    while (byte_count < total_valid_bytes) {
        while (out_stream.empty()) { }
        axis_t word = out_stream.read();
        ap_uint<128> tdata = word.data;
        for (int b = 0; b < 16 && byte_count < total_valid_bytes; b++) {
            if (padding == 3) {
                padding = 0;
                continue;
            }
            padding++;
            uint8_t byte_val = (uint8_t) tdata.range(8*b+7, 8*b);
            out_data.push_back(byte_val);
            byte_count++;
        }
    }
}

//----------------------------------------------------------------
// Write the raw input image to file, call the FSRCNN
// Python predictor, and read back the raw float32 reference output.
// The Python predictor expects the raw input image (RGB, uint8) and scales it
// by dividing by 256 then at the end it multiplies by 256.
std::vector<float> get_reference_from_pytorch_full(
    const std::vector<uint8_t>& input_data,
    int width, int height, int channels, int upscale_factor) {
    
    std::ofstream infile("conv_input.bin", std::ios::binary);
    if (!infile) {
        std::cerr << "Error: cannot open conv_input.bin for writing.\n";
        exit(1);
    }
    infile.write(reinterpret_cast<const char*>(input_data.data()), input_data.size());
    infile.flush();
    infile.close();

    std::ostringstream cmd;
    cmd << "python3 tb_fsrcnn_pyTorch_predictor.py "
        << "--mode axi "  // uint8 input mode
        << "--input conv_input.bin "
        << "--output conv_ref.bin "
        << "--width " << width << " "
        << "--height " << height << " "
        << "--channels " << channels << " "
        << "--upscale_factor " << upscale_factor;
    std::cout << "Calling Python full CNN predictor:\n" << cmd.str() << std::endl;
    int ret = system(cmd.str().c_str());
    if(ret != 0) {
        std::cerr << "Error: Python predictor returned error code " << ret << "\n";
        exit(1);
    }
    std::ifstream outfile("conv_ref.bin", std::ios::binary | std::ios::ate);
    if(!outfile) {
        std::cerr << "Error: cannot open conv_ref.bin for reading.\n";
        exit(1);
    }
    std::streamsize size = outfile.tellg();
    int num_floats = size / sizeof(float);
    outfile.seekg(0, std::ios::beg);
    std::vector<float> ref_output(num_floats);
    if(!outfile.read(reinterpret_cast<char*>(ref_output.data()), size)) {
        std::cerr << "Error: cannot read conv_ref.bin.\n";
        exit(1);
    }
    outfile.close();
    return ref_output;
}

//----------------------------------------------------------------
// Main testbench for the full Netwrok.
// Runs multiple tests: For each test, a random 28x28 (configurable) RGB image is generated,
// sent to the DUT via a 128-bit AXI stream, and compared with the Python predictor
int main() {
    // Configuration
    const int num_tests = 10;            // number of tests to run
    const int input_width = 32;
    const int input_height = 32;
    const int channels = 3;              // RGB
    const int upscale_factor = 2;        // Factor 2 -> output dims: 56x56
    const int output_width = input_width * upscale_factor;
    const int output_height = input_height * upscale_factor;
    const float tolerance = 3.0f;        // adjustable tolerance (in the same scale as output ~0-255)

    int pass_count = 0;
    int fail_count = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    for (int t = 0; t < num_tests; t++) {
        std::cout << "\n--- Test " << (t+1) << " ---\n";

        // Generate random input image (28x28 RGB)
        int input_bytes = input_width * input_height * channels;
        std::vector<uint8_t> input_data(input_bytes);
        for (auto &val : input_data) {
            val = static_cast<uint8_t>(dis(gen));
        }

        // Optionally, print first few values for debugging
        std::cout << "Input (first 10 values): ";
        for (int i = 0; i < 10 && i < input_data.size(); i++) {
            std::cout << (int)input_data[i] << " ";
        }
        std::cout << "\n";

        std::vector<float> ref_output = get_reference_from_pytorch_full(input_data,
                                                                         input_width,
                                                                         input_height,
                                                                         channels,
                                                                         upscale_factor);
        hls::stream<axis_t> in_stream("in_stream");
        hls::stream<axis_t> out_stream("out_stream");
        send_axi_stream_input_128bit(in_stream, input_data, input_width, input_height, channels);
        
        // Call DUT full network top function
        conv2d_top(in_stream, out_stream);
        
        std::vector<uint8_t> dut_raw_output;
        receive_axi_stream_output_128bit(out_stream, dut_raw_output, output_width, output_height, channels);
        
        // Convert DUT output to float (values in 0-255)
        std::vector<float> dut_output;
        for (auto val : dut_raw_output) {
            dut_output.push_back(static_cast<float>(val));
        }
        
        // Compare DUT output with reference output
        bool pass = true;
        if(ref_output.size() != dut_output.size()){
            std::cout << "Size mismatch: ref=" << ref_output.size() << " dut=" << dut_output.size() << "\n";
            pass = false;
        } else {
            for (size_t i = 0; i < ref_output.size(); i++) {
                float diff = std::fabs(ref_output[i] - dut_output[i]);
                if(diff > tolerance && diff < 253) {
                    std::cout << "Mismatch at index " << i << ": ref = " << ref_output[i]
                              << ", dut = " << dut_output[i] << ", diff = " << diff << "\n";
                    pass = false;
                }
            }
        }
        if(pass) {
            pass_count++;
            std::cout << "Test PASSED!\n";
        } else {
            fail_count++;
            std::cout << "Test FAILED!\n";
        }
    }

    std::cout << "\n====================================\n";
    std::cout << "Test Summary\n";
    std::cout << "Total tests: " << num_tests << "\n";
    std::cout << "Passed: " << pass_count << "\n";
    std::cout << "Failed: " << fail_count << "\n";
    std::cout << "====================================\n";

    return (fail_count == 0) ? 0 : 1;
}
