#include "bilinear_interpolation_byte_v2.h"
#include "image_coin_tile.h"
#include <cstdlib>


int main() {
    // Declare input and output streams
    hls::stream<axis_t> in_stream("input_stream");
    hls::stream<axis_t> out_stream("output_stream");


    // Define the number of test values
    //const int TEST_SIZE = 16;

    pixel_t test_data[NUM_TRANSFERS + 1];


    for(int i = 0; i < NUM_TRANSFERS; i++){
    	pixel_t temp = coin_tile_low_res[i];
    	test_data[i] = temp;
    }

    test_data[NUM_TRANSFERS] = (pixel_t)0;

	// Fill the input stream with test data
	//for (int i = 0; i < NUM_TRANSFERS; i++) {
    for (int i = 0; i < NUM_TRANSFERS+1; i++) {
		axis_t input_stream;
		input_stream.data = test_data[i];
		//input_stream.last = (i == NUM_TRANSFERS - 1); // Set the last signal for the last element
		input_stream.last = (i == NUM_TRANSFERS); // Set the last signal for the last element
		input_stream.keep = 0b1;
		input_stream.strb = 0b1;
		in_stream.write(input_stream);
	}


    // Call the HLS function
    bilinear_interpolation_byte(in_stream, out_stream);

    // Check the output stream
    bool success = false;

    int j = 0;

	//make sure the correct number of transfers are passed in
	while(j < NUM_TRANSFERS_OUT){

		while(!out_stream.empty()){

			// Read the output stream
			axis_t output_element = out_stream.read();
			pixel_t data = output_element.data;

			// Verify the data matches
			if ((uint8_t)data != coin_tile_interpolated[j]) {
				std::cout << "ERROR: Mismatch at index " << j
						  << " (expected " << (int)coin_tile_interpolated[j]
						  << ", got " << data << ")\n";
				success = false;
			}
			else {
				std::cout << "SUCCESS: Match at index " << j
						  << " (expected " << coin_tile_interpolated[j]
						  << ", got " << data << ")\n";
				success = true;
			}

			if(output_element.keep != 0b1 || output_element.strb != 0b1){
				success = false;
				std:: cout << "keep or strb wrong";
				break;
			}


			if((j < (NUM_TRANSFERS_OUT - 1)) && output_element.last == true){
				success = false;
				std::cout << "last triggered before end\n";
				break;
			}

			else if(j == (NUM_TRANSFERS_OUT - 1) && output_element.last == false){
				success = false;
				std::cout << "last not triggered at end\n";
				break;
			}

			j++;

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
