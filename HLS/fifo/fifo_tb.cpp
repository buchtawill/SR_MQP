#include "fifo.h"


// Testbench to validate the BilinearInterpolation function
int main() {
    // Declare input and output streams
    hls::stream<axis_t> in_stream;
    hls::stream<axis_t> out_stream;

    // Generate a test image (28x28x3) and push to the input stream
    for (int row = 0; row < HEIGHT_IN; row++) {
        for (int col = 0; col < WIDTH_IN; col++) {
            for (int ch = 0; ch < CHANNELS; ch++) {
                axis_t input_data;
                input_data.data = row + col + ch; // Example: simple pattern for testing
                input_data.last = (row == HEIGHT_IN - 1 && col == WIDTH_IN - 1 && ch == CHANNELS - 1);
                in_stream.write(input_data);
            }
        }
    }

    // Call the DUT (Device Under Test)
    fifo(in_stream, out_stream);

    // Verify the output
    bool success = false;
    int total_output_elements = HEIGHT_IN * WIDTH_IN * CHANNELS;
    int elements_read = 0;

    while (elements_read < total_output_elements) {
        std::cerr << "Getting in while loop";
        if (!out_stream.empty()) { // Check if the output stream is not empty
        	std::cerr << "Reading from stream";
            axis_t output_data = out_stream.read(); // Read data from the output stream

            // Compute expected value
            int row = elements_read / (WIDTH_IN * CHANNELS);
            int col = (elements_read / CHANNELS) % WIDTH_IN;
            int ch = elements_read % CHANNELS;
            pixel_t expected_value = row + col + ch;

            // Check the data
            if (output_data.data != expected_value) {
                std::cerr << "Mismatch at element " << elements_read
                          << " [row=" << row << ", col=" << col << ", ch=" << ch
                          << "]: Expected " << (int)expected_value
                          << ", Got " << (int)output_data.data << std::endl;
                success = false;
            }
            else{
            	/*
                std::cerr << "Match at element " << elements_read
                          << " [row=" << row << ", col=" << col << ", ch=" << ch
                          << "]: Expected " << (int)expected_value
                          << ", Got " << (int)output_data.data << std::endl; */
                success = true;
            }

            // Check if the "last" signal is correct for the last value
            if (elements_read == total_output_elements - 1) {
                if (!output_data.last) {
                    std::cerr << "Error: 'last' signal is not set for the last element.\n";
                    success = false;
                }
            } else {
                if (output_data.last) {
                    std::cerr << "Error: 'last' signal is incorrectly set for a non-last element.\n";
                    success = false;
                }
            }

            elements_read++; // Increment the count of elements read
        } else {
            // Output stream is empty but we expect more data
            std::cerr << "Error: Output stream is empty before all values are read.\n";
            success = false;
            break;
        }
    }

    // Final result
    if (success) {
        std::cout << "Test passed successfully!" << std::endl;
    } else {
        std::cerr << "Test failed." << std::endl;
        return 1;
    }

    return 0;
}
