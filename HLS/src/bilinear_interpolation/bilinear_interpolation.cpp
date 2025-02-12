#include "bilinear_interpolation.h"
// HLS function to read from input stream and write to output stream
#include <ap_fixed.h>
#include <algorithm>

int bilinear_interpolation_calculations(pixel_t image_in[HEIGHT_IN * WIDTH_IN * CHANNELS],
                                        pixel_t image_out[HEIGHT_OUT * WIDTH_OUT * CHANNELS]) {

    fixed widthRatio  = fixed(WIDTH_IN - 1) / fixed(WIDTH_OUT - 1);
    fixed heightRatio = fixed(HEIGHT_IN - 1) / fixed(HEIGHT_OUT - 1);

    for (int y_out = 0; y_out < HEIGHT_OUT; ++y_out) {

        #pragma HLS PIPELINE II=11

        for (int x_out = 0; x_out < WIDTH_OUT; ++x_out) {

            #pragma HLS UNROLL factor=2

            // Compute the corresponding input coordinates in fixed point
            fixed x_in = x_out * widthRatio;
            fixed y_in = y_out * heightRatio;

            // Determine the four nearest neighbors
            int x0 = static_cast<int>(x_in);
            int y0 = static_cast<int>(y_in);
            int x1 = std::min(x0 + 1, WIDTH_IN - 1);
            int y1 = std::min(y0 + 1, HEIGHT_IN - 1);

            // Calculate the interpolation weights using fixed-point
            fixed dx = x_in - fixed(x0);
            fixed dy = y_in - fixed(y0);

            fixed w00 = (fixed(1) - dx) * (fixed(1) - dy);
            fixed w10 = dx * (fixed(1) - dy);
            fixed w01 = (fixed(1) - dx) * dy;
            fixed w11 = dx * dy;

            // Perform bilinear interpolation for each channel
            for (int ch = 0; ch < CHANNELS; ++ch) {

                #pragma HLS UNROLL factor=2

                int index00 = (y0 * WIDTH_IN + x0) * CHANNELS + ch;
                int index10 = (y0 * WIDTH_IN + x1) * CHANNELS + ch;
                int index01 = (y1 * WIDTH_IN + x0) * CHANNELS + ch;
                int index11 = (y1 * WIDTH_IN + x1) * CHANNELS + ch;

                // Compute interpolated value using fixed-point
                fixed interpolated_value =
                    w00 * fixed(image_in[index00]) +
                    w10 * fixed(image_in[index10]) +
                    w01 * fixed(image_in[index01]) +
                    w11 * fixed(image_in[index11]);

                // Assign the interpolated value to the output image (rounding correctly)
                int out_index = (y_out * WIDTH_OUT + x_out) * CHANNELS + ch;
                image_out[out_index] = static_cast<pixel_t>(interpolated_value + fixed(0.5)); // Rounding
            }
        }
    }

    return 1;
}


void bilinear_interpolation(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return
	//#pragma HLS INTERFACE ap_ctrl_hs port=return

	pixel_t input_data_stored[PIXELS_IN];
	#pragma HLS BIND_STORAGE variable=input_data_stored type=RAM_1P impl=URAM

	pixel_t output_data_stored[PIXELS_OUT];
	#pragma HLS BIND_STORAGE variable=input_data_stored type=RAM_1P impl=URAM


	int i = 0;

	/*
	//update later with reset signal, but make sure FIFO is cleared on start up
	if(i == NUM_TRANSFERS || i == 0){
		for(int k = 0; k < NUM_TRANSFERS; k++){
			input_data_stored[k] = 0;
			//input_last_stored[k] = 0;
			//input_keep_stored[k] = 0;
		}
	} */


    pixel_t unloaded[PIXELS_IN];


	//make sure the correct number of transfers are passed in
	while(i < NUM_TRANSFERS){

		while(!in_stream.empty()){

			//if the correct number of transfers have been received stop taking in new data
			if(i == NUM_TRANSFERS){
				break;
			}

			axis_t temp_input = in_stream.read();
	        data_streamed temp_data = temp_input.data;

			int upper_range = 0;
			int lower_range = 0;
			uint8_t temp_pixel;

			for(int transfer_pixel = 0; transfer_pixel < PIXELS_PER_TRANSFER; transfer_pixel++){
				upper_range = transfer_pixel * BITS_PER_PIXEL + 7;
				lower_range = transfer_pixel * BITS_PER_PIXEL;
				temp_pixel = temp_data.range(upper_range, lower_range);

				unloaded[i * 16 + transfer_pixel] = temp_pixel;
			}

			i++;
		}
	}


	bilinear_interpolation_calculations(unloaded, output_data_stored);

    data_streamed output_loaded[NUM_TRANSFERS_OUT];

    int k = 0;

    if(i >= NUM_TRANSFERS){
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

			int upper_range = 0;
			int lower_range = 0;
			data_streamed temp_load;

			for(int transfer_pixel = 0; transfer_pixel < PIXELS_PER_TRANSFER; transfer_pixel++){
				upper_range = transfer_pixel * BITS_PER_PIXEL + 7;
				lower_range = transfer_pixel * BITS_PER_PIXEL;
				temp_load.range(upper_range, lower_range) = output_data_stored[k * PIXELS_PER_TRANSFER + transfer_pixel];
			}

			temp_output.data = temp_load;
			temp_output.last = last;
			temp_output.keep = 0xFFFF;
			temp_output.strb = 0xFFFF;

			// Write data to output stream
			out_stream.write(temp_output);

            k++;

    	}
    }

}
