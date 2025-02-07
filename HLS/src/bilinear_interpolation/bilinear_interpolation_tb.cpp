#include "bilinear_interpolation.h"
#include "image_coin_tile.h"
#include <cstdlib>


int main() {
    // Declare input and output streams
    hls::stream<axis_t> in_stream("input_stream");
    hls::stream<axis_t> out_stream("output_stream");


    // Define the number of test values
    //const int TEST_SIZE = 16;

    data_streamed test_data[NUM_TRANSFERS];


    /*
    for(int i = 0; i < NUM_TRANSFERS; i++){
    	pixel_t temp = coin_tile_low_res[i];
    	test_data[i] = temp;
    } */

    temp_streamed loaded[147];

    for(int load = 0; load < 147; load++){
    	int upper_range = 0;
    	int lower_range = 0;
    	temp_streamed temp_load;

    	for(int transfer_pixel = 0; transfer_pixel < 16; transfer_pixel++){
    		upper_range = transfer_pixel * 8 + 7;
    		lower_range = transfer_pixel * 8;
    		temp_load.range(upper_range, lower_range) = coin_tile_low_res[load * 16 + transfer_pixel];
    	}

    	loaded[load] = temp_load;
    }

    for(int unload = 0; unload < 147; unload++){
    	int upper_range = 0;
    	int lower_range = 0;
    	uint8_t temp_pixel;

    	for(int transfer_pixel = 0; transfer_pixel < 16; transfer_pixel++){
    		upper_range = transfer_pixel * 8 + 7;
    		lower_range = transfer_pixel * 8;
    		temp_pixel = loaded[unload].range(upper_range, lower_range);

    		test_data[unload * 16 + transfer_pixel] = (data_streamed)temp_pixel;
    		/*
			if ((uint8_t)temp_pixel != coin_tile_low_res[unload * 16 + transfer_pixel]) {
				std::cout << "ERROR: Load Mismatch at index " << (unload * 16 + transfer_pixel)
						  << " (expected " << (int)coin_tile_low_res[unload * 16 + transfer_pixel]
						  << ", got " << (int)temp_pixel << ")\n";
			}
			else {
				std::cout << "SUCCESS: Load Match at index " << (unload * 16 + transfer_pixel)
						  << " (expected " << coin_tile_low_res[unload * 16 + transfer_pixel]
						  << ", got " << (int)temp_pixel << ")\n";
			} */
    	}


    }

	// Fill the input stream with test data
	//for (int i = 0; i < NUM_TRANSFERS; i++) {
    for (int i = 0; i < NUM_TRANSFERS; i++) {
		axis_t input_stream;
		input_stream.data = test_data[i];
		//input_stream.last = (i == NUM_TRANSFERS - 1); // Set the last signal for the last element
		input_stream.last = (i == NUM_TRANSFERS - 1); // Set the last signal for the last element
		input_stream.keep = 0b1;
		input_stream.strb = 0b1;
		in_stream.write(input_stream);
	}


    //Instantiate DUT
    bilinear_interpolation(in_stream, out_stream);

    // Check the output stream
    bool success = false;

    int j = 0;

	//make sure the correct number of transfers are passed in
	while(j < NUM_TRANSFERS_OUT){

		while(!out_stream.empty()){

			// Read the output stream
			axis_t output_element = out_stream.read();
			data_streamed data = output_element.data;



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
