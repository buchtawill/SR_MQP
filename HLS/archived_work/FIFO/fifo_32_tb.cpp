#include "fifo_32.h"


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

    for(int i = 0; i < (WIDTH_IN * HEIGHT_IN * CHANNELS); i+= 4){
        pixel_t pixel0 = i % 256;
        pixel_t pixel1 = (i + 1) % 256;
        pixel_t pixel2 = (i + 2) % 256;
        pixel_t pixel3 = (i + 3) % 256;
    }

    // Call the DUT (Device Under Test)
    fifo_32(in_stream, out_stream);

    // Verify the output
    bool success = false;
    int total_output_elements = HEIGHT_IN * WIDTH_IN * CHANNELS;
    int elements_read = 0;

    while (elements_read < total_output_elements) {
    //std::cerr << "Getting in while loop";
    if (!out_stream.empty()) { // Check if the output stream is not empty
        axis_t output_data = out_stream.read(); // Read data from the output stream
        //std::cerr << "Reading from stream: " << (int)output_data.data << std::endl;

        // Unpack 4 pixels from the 32-bit output_data.data
        for (int i = 0; i < 4; i++) {
            if (elements_read >= total_output_elements) break; // Stop if all elements are read

            pixel_t pixel = (output_data.data >> (i * 8)) & 0xFF;
            //std::cerr << "data section" << (int)pixel << std::endl;

            // Compute expected value
            int row = elements_read / (WIDTH_IN * CHANNELS);
            int col = (elements_read / CHANNELS) % WIDTH_IN;
            int ch = elements_read % CHANNELS;
            pixel_t expected_value = row + col + ch;

            // Check the data
            if (pixel != expected_value) {
                std::cerr << "Mismatch at element " << elements_read
                          << " [row=" << row << ", col=" << col << ", ch=" << ch
                          << "]: Expected " << (int)expected_value
                          << ", Got " << (int)pixel << std::endl;
                success = false;
            } else {
                /*
                std::cerr << "Match at element " << elements_read
                          << " [row=" << row << ", col=" << col << ", ch=" << ch
                          << "]: Expected " << (int)expected_value
                          << ", Got " << (int)pixel << std::endl; */
                success = true;
            }

            elements_read++; // Increment the count of elements read
        }

        // Check if the "last" signal is correct for the last value
        if (elements_read == total_output_elements) {
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
