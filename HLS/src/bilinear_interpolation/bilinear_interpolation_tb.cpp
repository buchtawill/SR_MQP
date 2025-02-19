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

    data_streamed loaded[196]; //28 * 28 / 4

    for (int load = 0; load < 196; load++) {
        data_streamed temp_load = 0;

        int base_index = load * 4 * CHANNELS; //4 = # pixels/transfer, 3 = num

        for (int pixel_transfer = 0; pixel_transfer < 4; pixel_transfer++) {
            pixel_t R = coin_tile_low_res[base_index + pixel_transfer * 3];
            pixel_t G = coin_tile_low_res[base_index + pixel_transfer * 3 + 1];
            pixel_t B = coin_tile_low_res[base_index + pixel_transfer * 3 + 2];

            temp_load.range(pixel_transfer * 32 + 7, pixel_transfer * 32)     = R;
            temp_load.range(pixel_transfer * 32 + 15, pixel_transfer * 32 + 8) = G;
            temp_load.range(pixel_transfer * 32 + 23, pixel_transfer * 32 + 16) = B;
            //temp_load.range(pixel_transfer * 32 + 31, pixel_transfer * 32 + 24) = 0; // Don't-care bits
        }

        loaded[load] = temp_load;
    }


	// Fill the input stream with test data
	//for (int i = 0; i < NUM_TRANSFERS; i++) {
    for (int i = 0; i < 196; i++) {
		axis_t input_stream;
		input_stream.data = loaded[i];
		//input_stream.last = (i == NUM_TRANSFERS - 1); // Set the last signal for the last element
		input_stream.last = (i == 196 - 1); // Set the last signal for the last element
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
				if ((uint8_t)temp_pixel >= (coin_tile_interpolated[j * 16 + transfer_pixel] + 4)){
					std::cout << "ERROR: Mismatch at index " << (j * 16 + transfer_pixel)
							  << " (expected " << (int)coin_tile_interpolated[j * 16 + transfer_pixel]
							  << ", got " << (int)temp_pixel << ")\n";
					success = false;
				}
				else if ((uint8_t)temp_pixel <= (coin_tile_interpolated[j * 16 + transfer_pixel] - 4)){
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
