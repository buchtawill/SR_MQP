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

void stream_samples_in(hls::stream<axis_t> &in_stream, pixel_t input_data_stored[PIXELS_IN]){

	int i = 0;

	data_streamed input_streams_stored[NUM_TRANSFERS];

	//make sure the correct number of transfers are passed in
	while(i < NUM_TRANSFERS){

		while(!in_stream.empty()){

			//if the correct number of transfers have been received stop taking in new data
			if(i == NUM_TRANSFERS){
				break;
			}

			axis_t temp_input = in_stream.read();
			input_streams_stored[i] = temp_input.data;
	        //data_streamed temp_data = temp_input.data;


			i++;
		}
	}

    for (int transfer = 0; transfer < NUM_TRANSFERS; transfer++) {

        // Compute the base index for storing extracted values
        int base_index = transfer * PIXELS_PER_TRANSFER * CHANNELS;

        // Extract RGB values from {xbgr-xbgr-xbgr-xbgr} format
        for (int j = 0; j < PIXELS_PER_TRANSFER; j++) {
            input_data_stored[base_index + j * CHANNELS]     = input_streams_stored[transfer].range(j * BITS_PER_PIXEL + 7, j * BITS_PER_PIXEL);
            input_data_stored[base_index + j * CHANNELS + 1] = input_streams_stored[transfer].range(j * BITS_PER_PIXEL + 15, j * BITS_PER_PIXEL + 8);
            input_data_stored[base_index + j * CHANNELS + 2] = input_streams_stored[transfer].range(j * BITS_PER_PIXEL + 23, j * BITS_PER_PIXEL + 16);
            // Don't store X (bits 31:24)
        }
    }
}

void stream_samples_out(pixel_t output_data_stored[PIXELS_OUT], hls::stream<axis_t> &out_stream){

    data_streamed loaded[NUM_TRANSFERS_OUT];

    for (int load = 0; load < NUM_TRANSFERS_OUT; load++) {
        data_streamed temp_load = 0;

        int base_index = load * 4 * CHANNELS;

        for (int pixel_transfer = 0; pixel_transfer < 4; pixel_transfer++) {
            pixel_t R = output_data_stored[base_index + pixel_transfer * CHANNELS];
            pixel_t G = output_data_stored[base_index + pixel_transfer * CHANNELS + 1];
            pixel_t B = output_data_stored[base_index + pixel_transfer * CHANNELS + 2];

            temp_load.range(pixel_transfer * BITS_PER_PIXEL + 7, pixel_transfer * BITS_PER_PIXEL)     = R;
            temp_load.range(pixel_transfer * BITS_PER_PIXEL + 15, pixel_transfer * BITS_PER_PIXEL + 8) = G;
            temp_load.range(pixel_transfer * BITS_PER_PIXEL + 23, pixel_transfer * BITS_PER_PIXEL + 16) = B;
            //temp_load.range(pixel_transfer * 32 + 31, pixel_transfer * 32 + 24) = 0; // Don't-care bits
        }

        loaded[load] = temp_load;
    }


	//Fill the output stream with interpolated data
    for (int i = 0; i < NUM_TRANSFERS_OUT; i++) {
		axis_t output_stream;
		output_stream.data = loaded[i];
		output_stream.last = (i == NUM_TRANSFERS_OUT - 1); // Set the last signal for the last element
		output_stream.keep = 0xFFFF;
		output_stream.strb = 0xFFFF;
		out_stream.write(output_stream);
	}
}


void bilinear_interpolation(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return
	//#pragma HLS INTERFACE ap_ctrl_hs port=return

	pixel_t input_data_stored[PIXELS_IN];
	#pragma HLS BIND_STORAGE variable=input_data_stored type=RAM_2P impl=BRAM

	pixel_t output_data_stored[PIXELS_OUT];
	#pragma HLS BIND_STORAGE variable=input_data_stored type=RAM_2P impl=BRAM


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


	#pragma HLS DATAFLOW
	stream_samples_in(in_stream, input_data_stored);

	bilinear_interpolation_calculations(input_data_stored, output_data_stored);

	stream_samples_out(output_data_stored, out_stream);


}
