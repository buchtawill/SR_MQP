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
#include "ap_fixed.h"
#include "conv2d.h"
#include "../image_coin_tile.h"

//----------------------------------------------------------------
// Reference generator for FIFO-array mode.
// This function writes a vector of float32 values (in channel–wise order) to file,
// flushes the output, calls the Python reference script (which uses "--mode fifo"),
// and then reads back the raw float32 reference output for comparison
std::vector<float> get_reference_from_pytorch_fifo(
    const std::vector<float>& input_data_float,
    int in_channels,
    int out_channels,
    int kernel_size,
    int width,
    int height,
    int out_width,
    int out_height,
    const std::string &conv_name,
    int stride,
    int padding)
{
    std::ofstream infile("conv_input.bin", std::ios::binary);
    if (!infile) {
        std::cerr << "Error: could not open conv_input.bin for writing.\n";
        exit(1);
    }
    infile.write(reinterpret_cast<const char*>(input_data_float.data()), input_data_float.size()*sizeof(float));
    infile.flush();
    infile.close();

    std::ostringstream cmd;
    cmd << "python3 tb_conv_pyTorch_predictor.py "
        << "--mode fifo "  // fifo mode: input is float32
        << "--input conv_input.bin "
        << "--output conv_ref.bin "
        << "--in_channels " << in_channels << " "
        << "--out_channels " << out_channels << " "
        << "--kernel_size " << kernel_size << " "
        << "--width " << width << " "
        << "--height " << height << " "
        << "--conv_name " << conv_name << " "
        << "--stride " << stride << " "
        << "--padding " << padding;
    std::cout << "Calling Python reference script (fifo mode):\n" << cmd.str() << std::endl;
    int ret = system(cmd.str().c_str());
    if(ret != 0) {
        std::cerr << "Error: Python script returned error code " << ret << "\n";
        exit(1);
    }
    std::ifstream outfile("conv_ref.bin", std::ios::binary | std::ios::ate);
    if (!outfile) {
        std::cerr << "Error: could not open conv_ref.bin for reading.\n";
        exit(1);
    }
    std::streamsize size = outfile.tellg();
    int num_floats = size / sizeof(float);
    outfile.seekg(0, std::ios::beg);
    std::vector<float> ref_output(num_floats);
    if (!outfile.read(reinterpret_cast<char*>(ref_output.data()), size)) {
        std::cerr << "Error: could not read conv_ref.bin.\n";
        exit(1);
    }
    outfile.close();
    return ref_output;
}

//----------------------------------------------------------------
// Compare two float vectors within a tolerance.
void compare_outputs(const std::vector<float>& ref,
                     const std::vector<float>& dut,
                     int &pass_count,
                     int &fail_count,
                     float tolerance)
{
    if (ref.size() != dut.size()) {
        std::cout << "Size mismatch: Expected " << ref.size() << " values, got " << dut.size() << " values.\n";
        fail_count++;
        return;
    }
    bool passed = true;
    for (size_t i = 0; i < ref.size(); i++) {
        float diff = std::fabs(ref[i] - dut[i]);
        if (diff > tolerance) {
            std::cout << "Mismatch at index " << i << ": ref = " << ref[i]
                      << ", dut = " << dut[i] << ", diff = " << diff << "\n";
            passed = false;
        }
    }
    if (passed) {
        pass_count++;
        std::cout << "Test PASSED!\n";
    } else {
        fail_count++;
        std::cout << "Test FAILED!\n";
    }
}

//----------------------------------------------------------------
// Main testbench for Mode 2 (Direct FIFO Array Test)
// This testbench runs a configurable number of random tests.
// It supports generating input values either in [0,1] or in [-1,1].
int main()
{
    // ------------------ Configuration ------------------
    const int num_tests = 1;             // Number of tests to run (change as desired)
    const bool use_positive_range = false; // true => generate values in [0,1]; false => [-1,1]

    // Convolution parameters for the python predictor 
    const int IN_CHN = 44;    // Number of input FIFOs (channels)
    const int OUT_CHN = 3;   // Number of output FIFOs (channels)
    int kernel_size = 9;     // Kernel size (1x1)
    int stride = 2;
    int padding = 4;
    int width = 28;
    int height = 28;
    int out_width = width;   // With padding, output size is preserved
    int out_height = height;
    std::string conv_name = "deconv"; // Convolution layer name MAKE SURE TO CHANGE THE DUT FUNCTION NAME AS WELL
    float tolerance = 0.02f; // Tolerance for float comparison

    int pass_count = 0;
    int fail_count = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist_positive(0.0f, 1.0f);
    std::uniform_real_distribution<float> dist_bipolar(-1.0f, 1.0f);

    std::cout << "\n--- Running " << num_tests << " random stimulus tests for FIFO Array Mode ---\n";
    for (int t = 0; t < num_tests; t++) {
        std::cout << "\nTest " << (t+1) << ":\n";

        // Generate per-channel random input data
        std::vector<float> input_float(width*height*IN_CHN);
        for (int ch = 0; ch < IN_CHN; ch++) {
            for (int i = 0; i < width * height; i++) {
                if (use_positive_range)
                    input_float[IN_CHN*i+ch] = dist_positive(gen);
                else
                    input_float[IN_CHN*i+ch] = dist_bipolar(gen);
            }
        }

        ch_stream_t tile_in[IN_CHN];
        std::vector<float> input_data_fifo;
        for (int ch = 0; ch < IN_CHN; ch++) {
            for (int i = 0; i < width * height; i++) {
                fixed_4_8_t fixed_val = input_float[IN_CHN*i+ch]; // implicit conversion
                tile_in[ch].write(fixed_val);
                //input_data_fifo.push_back(input_float[IN_CHN*i+ch]);
            }
        }

        // Get reference output using FIFO mode (Python script expects float32 input)
        std::vector<float> ref_output = get_reference_from_pytorch_fifo(input_float,
                                                                         IN_CHN,
                                                                         OUT_CHN,
                                                                         kernel_size,
                                                                         width,
                                                                         height,
                                                                         out_width,
                                                                         out_height,
                                                                         conv_name,
                                                                         stride,
                                                                         padding);

        // Prepare output FIFOs (array of fifo) 
        upscaled_stream_t map_out[OUT_CHN]; // Uncomment for deconv - comment out for conv
        out_height = height * stride;       // Uncomment for deconv - comment out for conv
        out_width = width * stride;         // Uncomment for deconv - comment out for conv

        // ch_stream_t map_out[OUT_CHN];    // Uncomment for conv - comment out for deconv



        // Call the DUT: each convolution layer

        // conv_feature_extraction0(tile_in, map_out);
        // conv_shrink0(tile_in, map_out);
        // conv_map0(tile_in, map_out);
        // conv_expand0(tile_in, map_out);
        conv_deconv0(tile_in, map_out);



        // Read DUT outputs from each output FIFO and convert to float
        std::vector<float> dut_output;
        for (int ch = 0; ch < OUT_CHN; ch++) {
            for (int i = 0; i < out_width * out_height; i++) {
                fixed_4_8_t out_val = map_out[ch].read();
                dut_output.push_back(out_val.to_float());
            }
        }

        compare_outputs(ref_output, dut_output, pass_count, fail_count, tolerance);
    }

    std::cout << "\n====================================\n";
    std::cout << "          Test Summary\n";
    std::cout << "Total tests run: " << num_tests << "\n";
    std::cout << "Passed: " << pass_count << "\n";
    std::cout << "Failed: " << fail_count << "\n";
    std::cout << "====================================\n\n";

    return (fail_count == 0) ? 0 : 1;
}
