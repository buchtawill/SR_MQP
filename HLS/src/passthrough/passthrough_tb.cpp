#include <iostream>
#include "passthrough.h"

int main() {

    // Declare input and output streams
    hls::stream<axis_t> in_stream("input_stream");
    hls::stream<axis_t> out_stream("output_stream");


    // Define the number of test values
    const int TEST_SIZE = 16;


    // Input data
    bytes_streamed test_data[TEST_SIZE] = {
		0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10,
		0x11121314, 0x15161718, 0x191A1B1C, 0x1D1E1F20,
		0x21222324, 0x25262728, 0x292A2B2C, 0x2D2E2F30,
		0x31323334, 0x35363738, 0x393A3B3C, 0x3D3E3F40
	};


    // Fill the input stream with test data
    for (int i = 0; i < TEST_SIZE; i++) {
        axis_t input_element;
        input_element.data = test_data[i];
        input_element.last = (i == TEST_SIZE - 1); // Set the last signal for the last element
        in_stream.write(input_element);
    }


    // Call the HLS function
    passthrough(in_stream, out_stream);


    // Check the output stream
    bool success = false;


    for (int i = 0; i < TEST_SIZE; i++) {
        // Read the output stream
        axis_t output_element = out_stream.read();
        // Verify the data matches
        if (output_element.data != test_data[i]) {
            std::cout << "ERROR: Mismatch at index " << i
                      << " (expected " << test_data[i]
                      << ", got " << output_element.data << ")\n";
            success = false;
        }
        else {
        	success = true;
        }
    }


    // Final result
    if (success) {
        std::cout << "Test Passed: All output data matches input data.\n";
    } else {
        std::cout << "Test Failed: Output data does not match input data.\n";
    }
    return 0;
}
