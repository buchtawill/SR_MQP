#include "bilinear_interpolation_byte_v2.h"
// HLS function to read from input stream and write to output stream

//math for bilinear interpolation
int bilinear_interpolation_calculations(pixel_t image_in[NUM_TRANSFERS],
                           	   	   	   	pixel_t image_out[NUM_TRANSFERS_OUT]){

    // Declare image buffers in URAM
    static pixel_t image_in_split[HEIGHT_IN][WIDTH_IN][CHANNELS];  // Input image stored in URAM
	#pragma HLS BIND_STORAGE variable=image_in_split type=RAM_1P impl=URAM

    // Declare image buffers in URAM
    static pixel_t image_out_split[HEIGHT_IN][WIDTH_IN][CHANNELS];  // Input image stored in URAM
	#pragma HLS BIND_STORAGE variable=image_in_split type=RAM_1P impl=URAM

    // Local variables for indexing the 3D matrix
    int row = 0, col = 0, channel = 0;

    //split values into matrix for calculating
    for(int j = 0; j < NUM_TRANSFERS; j++){
		// Store in value
		image_in_split[row][col][channel] = image_in[j];
		channel++;
		if (channel == CHANNELS) {
			channel = 0;
			col++;
			if (col == WIDTH_IN) {
				col = 0;
				row++;
			}
		}
    }

    float widthRatio  = float(WIDTH_IN - 1) / float(WIDTH_OUT - 1);
    float heightRatio = float(HEIGHT_IN - 1) / float(HEIGHT_OUT - 1);

    for (int y_out = 0; y_out < HEIGHT_OUT; ++y_out) {
	//#pragma HLS PIPELINE II=4

        for (int x_out = 0; x_out < WIDTH_OUT; ++x_out) {


            // Compute the corresponding input coordinates
            float x_in = x_out * widthRatio;
            float y_in = y_out * heightRatio;

            // Determine the four nearest neighbors
            int x0 = static_cast<int>(x_in);
            int y0 = static_cast<int>(y_in);
            int x1 = std::min(x0 + 1, WIDTH_IN - 1);
            int y1 = std::min(y0 + 1, HEIGHT_IN - 1);

            // Calculate the interpolation weights
            float dx = x_in - x0;
            float dy = y_in - y0;

            float w00 = (1 - dx) * (1 - dy);
            float w10 = dx * (1 - dy);
            float w01 = (1 - dx) * dy;
            float w11 = dx * dy;

            // Perform bilinear interpolation for each channel
            for (int ch = 0; ch < CHANNELS; ++ch) {
                float interpolated_value =
                    w00 * image_in_split[y0][x0][ch] +
                    w10 * image_in_split[y0][x1][ch] +
                    w01 * image_in_split[y1][x0][ch] +
                    w11 * image_in_split[y1][x1][ch];

                // Assign the interpolated value to the output image
                image_out_split[y_out][x_out][ch] = static_cast<uint8_t>(std::round(interpolated_value));
            }
        }
    }

    int row_out = 0, col_out = 0, channel_out = 0;

    //combine values into 1D matrix for transfer out
    for(int j = 0; j < NUM_TRANSFERS_OUT; j++){
		// Store in value
    	image_out[j] = image_out_split[row_out][col_out][channel_out];
		channel_out++;
		if (channel_out == CHANNELS) {
			channel_out = 0;
			col_out++;
			if (col_out == WIDTH_OUT) {
				col_out = 0;
				row_out++;
			}
		}
    }

    return 1;
}

void bilinear_interpolation_byte_v2(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return

	pixel_t input_data_stored[NUM_TRANSFERS];
	#pragma HLS BIND_STORAGE variable=input_data_stored type=RAM_1P impl=URAM

	pixel_t output_data_stored[NUM_TRANSFERS_OUT];
	#pragma HLS BIND_STORAGE variable=input_data_stored type=RAM_1P impl=URAM


	int i = 0;

	//update later with reset signal, but make sure FIFO is cleared on start up
	if(i == NUM_TRANSFERS || i == 0){
		for(int k = 0; k < NUM_TRANSFERS; k++){
			input_data_stored[k] = 0;
			//input_last_stored[k] = 0;
			//input_keep_stored[k] = 0;
		}
	}


	//make sure the correct number of transfers are passed in
	while(i < NUM_TRANSFERS){

		while(!in_stream.empty()){

			//if the correct number of transfers have been received stop taking in new data
			if(i == NUM_TRANSFERS){
				break;
			}

			axis_t temp_input = in_stream.read();
	        input_data_stored[i] = temp_input.data;
			i++;
		}
	}


	bilinear_interpolation_calculations(input_data_stored, output_data_stored);

	int k = 0;
    //once all the data has been read in
	//this might need to be NUM_TRANSFERS - 1
    if(i >= NUM_TRANSFERS){

        //transfer values from array
        while(k < NUM_TRANSFERS_OUT){

			//axis_t output_data = input_stored[j];
			axis_t temp_output;


			//ap_uint<BITS_PER_TRANSFER / 8> keep;
			bool last;

			if(k == (NUM_TRANSFERS_OUT - 1)){
				last = true;
			}
			else {
				last = false;
			}

			//don't change any of the signals that were passed in
			temp_output.data = output_data_stored[k];
			temp_output.last = last;
			temp_output.keep = 0b1;
			temp_output.strb = 0b1;


			// Write data to output stream
			out_stream.write(temp_output);

            k++;
        }
    }




    //reset array and allow new transfers
    if(i >= NUM_TRANSFERS && k >= NUM_TRANSFERS_OUT){

		for(int k = 0; k < NUM_TRANSFERS; k++){
			input_data_stored[k] = 0;
			//input_last_stored[k] = 0;
			//input_keep_stored[k] = 0;
		}

		i = 0;
		k = 0;
    }
}
