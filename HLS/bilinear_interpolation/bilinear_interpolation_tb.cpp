#include "bilinear_interpolation.h"

void bilinear_interpolation_tb_calcs(pixel_t image_in[HEIGHT_IN][WIDTH_IN][CHANNELS],
                           pixel_t image_out[HEIGHT_OUT][WIDTH_OUT][CHANNELS]) {

    // Perform Bilinear Interpolation
    for (int row_out = 0; row_out < HEIGHT_OUT; row_out++) {
        for (int col_out = 0; col_out < WIDTH_OUT; col_out++) {
            // Calculate corresponding position in the input image (floating point to allow interpolation)
            float row_in = row_out / (float)SCALE_FACTOR;
            float col_in = col_out / (float)SCALE_FACTOR;

            // Integer coordinates of the top-left pixel (floor values)
            int x1 = (int)row_in;
            int y1 = (int)col_in;

            // Fractional part of the coordinates (to calculate interpolation weights)
            float x_frac = row_in - x1;
            float y_frac = col_in - y1;

            // Clamping for boundary conditions (handling edges)
            int x2 = (x1 + 1) < HEIGHT_IN ? x1 + 1 : x1;
            int y2 = (y1 + 1) < WIDTH_IN ? y1 + 1 : y1;

            // Interpolation for each channel (e.g., RGB)
            for (int ch = 0; ch < CHANNELS; ch++) {
                // Interpolate horizontally first: between (x1, y1) and (x1, y2) for the top row,
                // and between (x2, y1) and (x2, y2) for the bottom row.
                float top_left = image_in[x1][y1][ch];
                float top_right = image_in[x1][y2][ch];
                float bottom_left = image_in[x2][y1][ch];
                float bottom_right = image_in[x2][y2][ch];

                // Perform bilinear interpolation in both horizontal and vertical directions
                float top_interp = (1 - y_frac) * top_left + y_frac * top_right; // Horizontal interpolation for top row
                float bottom_interp = (1 - y_frac) * bottom_left + y_frac * bottom_right; // Horizontal for bottom row

                // Now perform vertical interpolation between top_interp and bottom_interp
                float interpolated_value = (1 - x_frac) * top_interp + x_frac * bottom_interp;

                // Assign the interpolated value to the output image (clamping to pixel range)
                image_out[row_out][col_out][ch] = (pixel_t)(interpolated_value);
            }
        }
    }
}


// Testbench to validate the BilinearInterpolation function
void testbench() {
    hls::stream<axis_t> in_stream;
    hls::stream<axis_t> out_stream;

    static pixel_t in_temp[HEIGHT_IN][WIDTH_IN][CHANNELS];  // Input image stored in URAM
    static pixel_t out_temp[HEIGHT_OUT][WIDTH_OUT][CHANNELS]; // Output image buffer

    // Generate input image with a simple pattern
    for (int row = 0; row < HEIGHT_IN; row++) {
        for (int col = 0; col < WIDTH_IN; col++) {
            for (int ch = 0; ch < CHANNELS; ch++) {
                axis_t input_data;
                input_data.data = (row + col + ch) % 256;  // A simple test pattern
                input_data.last = (row == HEIGHT_IN - 1 && col == WIDTH_IN - 1 && ch == CHANNELS - 1);

                in_temp[row][col][ch] = (row + col + ch) % 256;
                in_stream.write(input_data);
            }
        }
    }

    // Call the BilinearInterpolation function
    bilinear_interpolation_v2(in_stream, out_stream);
    bilinear_interpolation_tb_calcs(in_temp, out_temp);



    // Check the output stream and compare with expected values
    bool test_passed = true;
    for (int row_out = 0; row_out < HEIGHT_OUT; row_out++) {
        for (int col_out = 0; col_out < WIDTH_OUT; col_out++) {
            for (int ch = 0; ch < CHANNELS; ch++) {
                axis_t output_data = out_stream.read();

                // Compare with expected value and flag if any mismatch
                if (output_data.data != out_temp[row_out][col_out][ch]) {
                    std::cout << "Mismatch at (" << row_out << ", " << col_out << ", " << ch << "): "
                              << "Expected " << out_temp[row_out][col_out][ch] << ", but got " << (int)output_data.data << std::endl;
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
