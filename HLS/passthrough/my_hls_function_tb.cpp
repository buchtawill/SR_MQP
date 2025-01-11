#include <iostream>
#include "my_hls_function.h"
#include <cstdlib>


int main() {
    // Declare input and output streams
    hls::stream<axis_t> in_stream("input_stream");
    hls::stream<axis_t> out_stream("output_stream");


    // Define the number of test values
    //const int TEST_SIZE = 16;

    bytes_streamed test_data[NUM_TRANSFERS];

    for(int i = 0; i < NUM_TRANSFERS; i++){
    	bytes_streamed temp = rand() % 256;
    	test_data[i] = temp;
    }

    /*
    // Input data
    bytes_streamed test_data[TEST_SIZE] = {
    		0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0x10
	};*/

    // Fill the input stream with test data
    for (int i = 0; i < NUM_TRANSFERS; i++) {
        axis_t input_element;
        input_element.data = test_data[i];
        input_element.last = (i == NUM_TRANSFERS - 1); // Set the last signal for the last element
        in_stream.write(input_element);
    }


    // Call the HLS function
    my_hls_function(in_stream, out_stream);

    // Check the output stream
    bool success = false;

    for (int i = 0; i < NUM_TRANSFERS; i++) {

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

        if(i < (NUM_TRANSFERS - 1) && output_element.last == true){
        	success = false;
        	break;
        }
        else if(i == (NUM_TRANSFERS - 1) && output_element.last == false){
        	success = false;
        	break;
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
