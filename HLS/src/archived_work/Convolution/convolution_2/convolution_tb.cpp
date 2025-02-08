#include "convolution.h"
#include <vector>
#include <cstdlib>

void convolution_tb(std::vector<std::vector<std::vector<pixel_t>>>& input_matrix,
                    std::vector<std::vector<std::vector<pixel_t>>>& kernel,
                    std::vector<std::vector<std::vector<pixel_t>>>& output_matrix) {

    // Step 1: Perform convolution (element-wise multiplication)
    for (int row = 0; row < MATRIX_SIZE; ++row) {
        for (int col = 0; col < MATRIX_SIZE; ++col) {
            for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
                pixel_t result = 0;

                // Iterate over the kernel
                for (int k_row = 0; k_row < KERNEL_SIZE; ++k_row) {
                    for (int k_col = 0; k_col < KERNEL_SIZE; ++k_col) {

                        int in_row = row + k_row - KERNEL_SIZE / 2;
                        int in_col = col + k_col - KERNEL_SIZE / 2;

                        // Boundary checking
                        if (in_row >= 0 && in_row < MATRIX_SIZE && in_col >= 0 && in_col < MATRIX_SIZE) {
                            result += input_matrix[in_row][in_col][ch] * kernel[k_row][k_col][ch];
                        }
                    }
                }

                output_matrix[row][col][ch] = result;
            }
        }
    }
}

// Testbench for the convolution
int main() {
    // Step 1: Declare the input, kernel, and expected output matrices as std::vector with row, col, channels
    std::vector<std::vector<std::vector<pixel_t>>> input_matrix(MATRIX_SIZE, std::vector<std::vector<pixel_t>>(MATRIX_SIZE, std::vector<pixel_t>(NUM_CHANNELS, 0)));

    for(int row = 0; row < MATRIX_SIZE; row++){
        for(int col = 0; col < MATRIX_SIZE; col++){
            for(int ch = 0; ch < NUM_CHANNELS; ch++){
                input_matrix[row][col][ch] = rand() % 255;
            }
        }
    }

    std::vector<std::vector<std::vector<pixel_t>>> kernel(KERNEL_SIZE, std::vector<std::vector<pixel_t>>(KERNEL_SIZE, std::vector<pixel_t>(NUM_CHANNELS, 0)));
    kernel = {
        {
            {1, 0, -1},
            {2, 0, -2},
            {1, 0, -1}
        },
        // Add kernels for other channels
        std::vector<std::vector<pixel_t>>(KERNEL_SIZE, std::vector<pixel_t>(NUM_CHANNELS, 1)),
        std::vector<std::vector<pixel_t>>(KERNEL_SIZE, std::vector<pixel_t>(NUM_CHANNELS, -1))
    };

    // Step 2: Initialize AXI-Stream interfaces
    hls::stream<axi_stream> in_stream;
    hls::stream<axi_stream> out_stream;
    hls::stream<axi_lite> kernel_stream;

    // Feed the input matrix into in_stream
    for (int row = 0; row < MATRIX_SIZE; row++) {
        for (int col = 0; col < MATRIX_SIZE; col++) {
            for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                axi_stream input_data;
                input_data.data = input_matrix[row][col][ch];
                input_data.last = (row == MATRIX_SIZE - 1 && col == MATRIX_SIZE - 1 && ch == NUM_CHANNELS - 1);  // Set last signal for last element
                in_stream.write(input_data);
            }
        }
    }

    // Feed the kernel matrix into kernel_stream
    for (int k_row = 0; k_row < KERNEL_SIZE; ++k_row) {
        for (int k_col = 0; k_col < KERNEL_SIZE; ++k_col) {
            for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
                axi_lite kernel_data;
                kernel_data.data = kernel[k_row][k_col][ch];
                kernel_stream.write(kernel_data);
            }
        }
    }

    // Step 3: Initialize output matrix and expected output matrix
    std::vector<std::vector<std::vector<pixel_t>>> expected_output(MATRIX_SIZE, std::vector<std::vector<pixel_t>>(MATRIX_SIZE, std::vector<pixel_t>(NUM_CHANNELS, 0)));

    // Call the convolution function (test the convolution)
    convolution_tb(input_matrix, kernel, expected_output);

    // Step 4: Call the top-level convolution function (simulate hardware function)
    convolution_top(in_stream, out_stream, kernel_stream);

    // Verify the output by reading values from out_stream
    bool pass = true;
    for (int row = 0; row < MATRIX_SIZE; ++row) {
        for (int col = 0; col < MATRIX_SIZE; ++col) {
            for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
                axi_stream output_data = out_stream.read();
                if (output_data.data != expected_output[row][col][ch]) {
                    std::cout << "Mismatch at (" << row << ", " << col << ", " << ch << "): "
                              << "Expected " << (int)expected_output[row][col][ch]
                              << ", got " << (int)output_data.data << std::endl;
                    pass = false;
                }
            }
        }
    }

    // Step 5: Print the final result
    if (pass) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
    }

    return pass ? 0 : 3; // Return 0 for pass, 3 for failure
}
