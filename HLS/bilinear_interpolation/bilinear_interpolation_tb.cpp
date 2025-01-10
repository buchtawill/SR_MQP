#include "bilinear_interpolation.h"

// Testbench to validate the BilinearInterpolation function
void testbench() {
    hls::stream<axis_t> in_stream;
    hls::stream<axis_t> out_stream;

    // Generate input image with a simple pattern
    for (int row = 0; row < HEIGHT_IN; row++) {
        for (int col = 0; col < WIDTH_IN; col++) {
            for (int ch = 0; ch < CHANNELS; ch++) {
                axis_t input_data;
                input_data.data = (row + col + ch) % 256;  // A simple test pattern
                input_data.last = (row == HEIGHT_IN - 1 && col == WIDTH_IN - 1 && ch == CHANNELS - 1);
                in_stream.write(input_data);
            }
        }
    }

    // Call the BilinearInterpolation function
    bilinear_interpolation(in_stream, out_stream);

    // Check the output stream and compare with expected values
    bool test_passed = true;
    for (int row_out = 0; row_out < HEIGHT_OUT; row_out++) {
        for (int col_out = 0; col_out < WIDTH_OUT; col_out++) {
            for (int ch = 0; ch < CHANNELS; ch++) {
                axis_t output_data = out_stream.read();

                // Calculate the expected value based on input
                // Since the interpolation is not really applied yet, we are just checking direct mapping
                int expected_value = ((row_out / SCALE_FACTOR) + (col_out / SCALE_FACTOR) + ch) % 256;

                // Compare with expected value and flag if any mismatch
                if (output_data.data != expected_value) {
                    std::cout << "Mismatch at (" << row_out << ", " << col_out << ", " << ch << "): "
                              << "Expected " << expected_value << ", but got " << (int)output_data.data << std::endl;
                    test_passed = false;
                }

                // Print the output pixel for debugging (can be commented out after testing)
                //std::cout << "Pixel (" << row_out << ", " << col_out << ", " << ch << ") = "
                //          << (int)output_data.data << (output_data.last ? " (last)" : "") << std::endl;
            }
        }
    }

    // Report the result of the test
    if (test_passed) {
        std::cout << "Test Passed: All output values are correct." << std::endl;
    } else {
        std::cout << "Test Failed: Some output values are incorrect." << std::endl;
    }
}

int main() {
    // Run the testbench
    testbench();
    return 0;
}
