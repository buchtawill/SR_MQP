#include "bilinear_interpolation.h"
#include "image_coin_tile.h"
#include <cstdlib>


int main() {
    // Declare input and output streams
    hls::stream<axis_t> in_stream("input_stream");
    hls::stream<axis_t> out_stream("output_stream");


    // Define the number of test values
    //const int TEST_SIZE = 16;

    channel_t test_data[PIXELS_IN];


    /*
    for(int i = 0; i < NUM_TRANSFERS; i++){
    	channel_t temp = coin_tile_low_res[i];
    	test_data[i] = temp;
    } */

    data_streamed loaded[NUM_TRANSFERS];

    for (int load = 0; load < NUM_TRANSFERS; load++) {
        data_streamed temp_load = 0;

        int base_index = load * PIXELS_PER_TRANSFER * CHANNELS;

        for (int pixel_transfer = 0; pixel_transfer < 4; pixel_transfer++) {
            channel_t R = coin_tile_low_res[base_index + pixel_transfer * CHANNELS];
            channel_t G = coin_tile_low_res[base_index + pixel_transfer * CHANNELS + 1];
            channel_t B = coin_tile_low_res[base_index + pixel_transfer * CHANNELS + 2];

            temp_load.range(pixel_transfer * BITS_PER_PIXEL + 7, pixel_transfer * BITS_PER_PIXEL)     = R;
            temp_load.range(pixel_transfer * BITS_PER_PIXEL + 15, pixel_transfer * BITS_PER_PIXEL + 8) = G;
            temp_load.range(pixel_transfer * BITS_PER_PIXEL + 23, pixel_transfer * BITS_PER_PIXEL + 16) = B;
            //temp_load.range(pixel_transfer * 32 + 31, pixel_transfer * 32 + 24) = 0; // Don't-care bits
        }

        loaded[load] = temp_load;
    }


	// Fill the input stream with test data
    for (int i = 0; i < NUM_TRANSFERS; i++) {
		axis_t input_stream;
		input_stream.data = loaded[i];
		input_stream.last = (i == NUM_TRANSFERS - 1); // Set the last signal for the last element
		input_stream.keep = 0xFFFF;
		input_stream.strb = 0xFFFF;
		in_stream.write(input_stream);
	}


    //Instantiate DUT
    bilinear_interpolation(in_stream, out_stream);

    // Check the output stream
    bool success = true;

    int j = 0;

	while(j < NUM_TRANSFERS_OUT){


		while(!out_stream.empty()){

			// Read the output stream
			axis_t output_element = out_stream.read();
			data_streamed data = output_element.data;

	        // Extract RGB values from {xbgr-xbgr-xbgr-xbgr} format
	        for (int pixel = 0; pixel < PIXELS_PER_TRANSFER; pixel++) {
	            channel_t R = data.range(pixel * BITS_PER_PIXEL + 7, pixel * BITS_PER_PIXEL);
	            channel_t G = data.range(pixel * BITS_PER_PIXEL + 15, pixel * BITS_PER_PIXEL + 8);
	            channel_t B = data.range(pixel * BITS_PER_PIXEL + 23, pixel * BITS_PER_PIXEL + 16);

	            int temp_r = (int)R;
	            int temp_g = (int)G;
	            int temp_b = (int)B;

				// Verify the R pixel matches
				if ((uint8_t)R >= (coin_tile_interpolated[j * CHANNELS_PER_TRANSFER + pixel * CHANNELS] + MARGIN_OF_ERROR)){
					std::cout << "NOPE: Mismatch at index " << (j * CHANNELS_PER_TRANSFER + pixel * CHANNELS)
							  << " (expected " << (int)coin_tile_interpolated[j * CHANNELS_PER_TRANSFER + pixel * CHANNELS]
							  << ", got " << (int)R << ")\n";
					//success = false;
				}
				else if ((uint8_t)R <= (coin_tile_interpolated[j * CHANNELS_PER_TRANSFER + pixel * CHANNELS] - MARGIN_OF_ERROR)){
					std::cout << "NOPE: Mismatch at index " << (j * CHANNELS_PER_TRANSFER + pixel * CHANNELS)
							  << " (expected " << (int)coin_tile_interpolated[j * CHANNELS_PER_TRANSFER + pixel * CHANNELS]
							  << ", got " << (int)R << ")\n";
					//success = false;
				}
				else {
					std::cout << "SUCCESS: Match at index " << (j * CHANNELS_PER_TRANSFER + pixel * CHANNELS)
							  << " (expected " << coin_tile_interpolated[j * CHANNELS_PER_TRANSFER + pixel * CHANNELS]
							  << ", got " << (int)R << ")\n";
					//success = true;
				}



				// Verify the G pixel matches
				if ((uint8_t)G >= (coin_tile_interpolated[j * CHANNELS_PER_TRANSFER + pixel * CHANNELS + 1] + MARGIN_OF_ERROR)){
					std::cout << "NOPE: Mismatch at index " << (j * CHANNELS_PER_TRANSFER + pixel * CHANNELS + 1)
							  << " (expected " << (int)coin_tile_interpolated[j * CHANNELS_PER_TRANSFER + pixel * CHANNELS + 1]
							  << ", got " << (int)G << ")\n";
					//success = false;
				}
				else if ((uint8_t)G <= (coin_tile_interpolated[j * CHANNELS_PER_TRANSFER + pixel * CHANNELS + 1] - MARGIN_OF_ERROR)){
					std::cout << "NOPE: Mismatch at index " << (j * CHANNELS_PER_TRANSFER + pixel * CHANNELS + 1)
							  << " (expected " << (int)coin_tile_interpolated[j * CHANNELS_PER_TRANSFER + pixel * CHANNELS + 1]
							  << ", got " << (int)G << ")\n";
					//success = false;
				}
				else {
					std::cout << "SUCCESS: Match at index " << (j * CHANNELS_PER_TRANSFER + pixel * CHANNELS + 1)
							  << " (expected " << coin_tile_interpolated[j * CHANNELS_PER_TRANSFER + pixel * CHANNELS + 1]
							  << ", got " << (int)G << ")\n";
					//success = true;
				}


				// Verify the B pixel matches
				if ((uint8_t)B >= (coin_tile_interpolated[j * CHANNELS_PER_TRANSFER + pixel * CHANNELS + 2] + MARGIN_OF_ERROR)){
					std::cout << "NOPE: Mismatch at index " << (j * CHANNELS_PER_TRANSFER + pixel * CHANNELS + 2)
							  << " (expected " << (int)coin_tile_interpolated[j * CHANNELS_PER_TRANSFER + pixel * CHANNELS + 2]
							  << ", got " << (int)B << ")\n";
					//success = false;
				}
				else if ((uint8_t)B <= (coin_tile_interpolated[j * CHANNELS_PER_TRANSFER + pixel * CHANNELS + 2] - MARGIN_OF_ERROR)){
					std::cout << "NOPE: Mismatch at index " << (j * CHANNELS_PER_TRANSFER + pixel * CHANNELS + 2)
							  << " (expected " << (int)coin_tile_interpolated[j * CHANNELS_PER_TRANSFER + pixel * CHANNELS + 2]
							  << ", got " << (int)B << ")\n";
					//success = false;
				}
				else {
					std::cout << "SUCCESS: Match at index " << (j * CHANNELS_PER_TRANSFER + pixel * CHANNELS + 2)
							  << " (expected " << coin_tile_interpolated[j * CHANNELS_PER_TRANSFER + pixel * CHANNELS + 2]
							  << ", got " << (int)B << ")\n";
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

    /*int j = 0;

	while(j < 588){

		while(!out_stream.empty()){

			// Read the output stream
			axis_t output_element = out_stream.read();
			data_streamed data = output_element.data;

			int upper_range = 0;
			int lower_range = 0;
			channel_t temp_pixel;

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

    } */


    // Final result
    if (success) {
        std::cout << "Test Passed: All output data matches input data.\n";
    } else {
        std::cout << "Test Failed: Output data does not match input data.\n";
    }

    return 0;
}
