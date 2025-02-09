#include "bilinear_interpolation.h"
#include "image_coin_tile.h"
#include <cstdlib>


int main() {
    // Declare input and output streams
    hls::stream<axis_t> in_stream("input_stream");
    hls::stream<axis_t> out_stream("output_stream");


    // Define the number of test values
    //const int TEST_SIZE = 16;

    pixel_t test_data[PIXELS_IN];


    /*
    for(int i = 0; i < NUM_TRANSFERS; i++){
    	pixel_t temp = coin_tile_low_res[i];
    	test_data[i] = temp;
    } */

    data_streamed loaded[NUM_TRANSFERS + 1];

    for(int load = 0; load < NUM_TRANSFERS; load++){
    	int upper_range = 0;
    	int lower_range = 0;
    	data_streamed temp_load;

    	for(int transfer_pixel = 0; transfer_pixel < 16; transfer_pixel++){
    		upper_range = transfer_pixel * 8 + 7;
    		lower_range = transfer_pixel * 8;
    		temp_load.range(upper_range, lower_range) = coin_tile_low_res[load * 16 + transfer_pixel];
    	}

    	loaded[load] = temp_load;
    }

    loaded[NUM_TRANSFERS] = (data_streamed)0;


	// Fill the input stream with test data
	//for (int i = 0; i < NUM_TRANSFERS; i++) {
    for (int i = 0; i < NUM_TRANSFERS+1; i++) {
		axis_t input_stream;
		input_stream.data = loaded[i];
		//input_stream.last = (i == NUM_TRANSFERS - 1); // Set the last signal for the last element
		input_stream.last = (i == NUM_TRANSFERS); // Set the last signal for the last element
		input_stream.keep = 0xFFFF;
		input_stream.strb = 0xFFFF;
		in_stream.write(input_stream);
	}


    //Instantiate DUT
    bilinear_interpolation(in_stream, out_stream);

    // Check the output stream
    bool success = true;

    int j = 0;

	while(j < 588){

		while(!out_stream.empty()){

			// Read the output stream
			axis_t output_element = out_stream.read();
			data_streamed data = output_element.data;

			int upper_range = 0;
			int lower_range = 0;
			pixel_t temp_pixel;

			for(int transfer_pixel = 0; transfer_pixel < 16; transfer_pixel++){
				upper_range = transfer_pixel * 8 + 7;
				lower_range = transfer_pixel * 8;
				temp_pixel = data.range(upper_range, lower_range);

				// Verify the data matches
				if ((uint8_t)temp_pixel != coin_tile_interpolated[j * 16 + transfer_pixel]) {
					std::cout << "ERROR: Mismatch at index " << (j * 16 + transfer_pixel)
							  << " (expected " << (int)coin_tile_interpolated[j * 16 + transfer_pixel]
							  << ", got " << (int)temp_pixel << ")\n";
					success = false;
				}
				else {
					std::cout << "SUCCESS: Match at index " << (j * 16 + transfer_pixel)
							  << " (expected " << coin_tile_interpolated[j * 16 + transfer_pixel]
							  << ", got " << (int)temp_pixel << ")\n";
					//success = true;
				}

			}



			if(output_element.keep != 0xFFFF || output_element.strb != 0xFFFF){
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
