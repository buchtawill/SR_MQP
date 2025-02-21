#include "bilinear_interpolation.h"
#include <ap_fixed.h>
#include <algorithm>

// HLS function for bilinear interpolation
int bilinear_interpolation_calculations(pixel_t image_in[SLIDER_HEIGHT_IN][SLIDER_WIDTH_IN],
                                        pixel_t image_out[SLIDER_HEIGHT_OUT][SLIDER_WIDTH_OUT]) {

    fixed widthRatio  = fixed(SLIDER_WIDTH_IN - 1) / fixed(SLIDER_WIDTH_OUT - 1);
    fixed heightRatio = fixed(SLIDER_HEIGHT_IN - 1) / fixed(SLIDER_HEIGHT_OUT - 1);

    for (int y_out = 0; y_out < SLIDER_HEIGHT_OUT; ++y_out) {

        #pragma HLS PIPELINE II=11

        for (int x_out = 0; x_out < SLIDER_WIDTH_OUT; ++x_out) {

            #pragma HLS UNROLL factor=2

            // Compute the corresponding input coordinates in fixed point
            fixed x_in = x_out * widthRatio;
            fixed y_in = y_out * heightRatio;

            // Determine the four nearest neighbors
            int x0 = static_cast<int>(x_in);
            int y0 = static_cast<int>(y_in);
            int x1 = std::min(x0 + 1, SLIDER_WIDTH_IN - 1);
            int y1 = std::min(y0 + 1, SLIDER_HEIGHT_IN - 1);

            // Calculate interpolation weights
            fixed dx = x_in - fixed(x0);
            fixed dy = y_in - fixed(y0);

            fixed w00 = (fixed(1) - dx) * (fixed(1) - dy);
            fixed w10 = dx * (fixed(1) - dy);
            fixed w01 = (fixed(1) - dx) * dy;
            fixed w11 = dx * dy;

            // Read pixel data from input (packed format 0xRRGGBB)
            pixel_t pixel00 = image_in[y0][x0];
            pixel_t pixel10 = image_in[y0][x1];
            pixel_t pixel01 = image_in[y1][x0];
            pixel_t pixel11 = image_in[y1][x1];

            // Extract RGB channels
            channel_t b00 = (pixel00 >> 16) & 0xFF, g00 = (pixel00 >> 8) & 0xFF, r00 = pixel00 & 0xFF;
            channel_t b10 = (pixel10 >> 16) & 0xFF, g10 = (pixel10 >> 8) & 0xFF, r10 = pixel10 & 0xFF;
            channel_t b01 = (pixel01 >> 16) & 0xFF, g01 = (pixel01 >> 8) & 0xFF, r01 = pixel01 & 0xFF;
            channel_t b11 = (pixel11 >> 16) & 0xFF, g11 = (pixel11 >> 8) & 0xFF, r11 = pixel11 & 0xFF;

            // Compute interpolated values for each channel
            fixed r_interp = w00 * fixed(r00) + w10 * fixed(r10) + w01 * fixed(r01) + w11 * fixed(r11);
            fixed g_interp = w00 * fixed(g00) + w10 * fixed(g10) + w01 * fixed(g01) + w11 * fixed(g11);
            fixed b_interp = w00 * fixed(b00) + w10 * fixed(b10) + w01 * fixed(b01) + w11 * fixed(b11);

            // Store interpolated values in `image_out` (rounded)
            pixel_t temp_pixel;
            temp_pixel.range(7, 0) = r_interp;
            temp_pixel.range(15, 8) = g_interp;
            temp_pixel.range(23, 16) = b_interp;

            image_out[y_out][x_out] = temp_pixel;
        }
    }

    return 1;
}

// Function to stream input samples into 2D array
void stream_samples_in(hls::stream<axis_t> &in_stream, pixel_t input_data_stored[HEIGHT_IN][WIDTH_IN]) {

    int i = 0;

    data_streamed input_streams_stored[NUM_TRANSFERS];

    while (i < NUM_TRANSFERS) {

        while (!in_stream.empty()) {

            if (i == NUM_TRANSFERS){
            	break;
            }

            axis_t temp_input = in_stream.read();
            input_streams_stored[i] = temp_input.data;

            i++;
        }
    }

    for (int transfer = 0; transfer < NUM_TRANSFERS; transfer++) {

        for (int j = 0; j < PIXELS_PER_TRANSFER; j++) {

            int index = transfer * PIXELS_PER_TRANSFER + j;
            int y = index / WIDTH_IN;
            int x = index % WIDTH_IN;
            input_data_stored[y][x] = input_streams_stored[transfer].range(j * BITS_PER_PIXEL + 23, j * BITS_PER_PIXEL);
        }
    }
}

// Function to stream output samples from 2D array
void stream_samples_out(pixel_t output_data_stored[HEIGHT_OUT][WIDTH_OUT], hls::stream<axis_t> &out_stream) {
    data_streamed loaded[NUM_TRANSFERS_OUT];

    for (int load = 0; load < NUM_TRANSFERS_OUT; load++) {

        data_streamed temp_load = 0;

        for (int pixel_transfer = 0; pixel_transfer < PIXELS_PER_TRANSFER; pixel_transfer++) {
            int index = load * PIXELS_PER_TRANSFER + pixel_transfer;
            int y = index / WIDTH_OUT;
            int x = index % WIDTH_OUT;

            temp_load.range(pixel_transfer * BITS_PER_PIXEL + 23, pixel_transfer * BITS_PER_PIXEL) = output_data_stored[y][x];
        }

        loaded[load] = temp_load;
    }

    for (int i = 0; i < NUM_TRANSFERS_OUT; i++) {
        axis_t output_stream;
        output_stream.data = loaded[i];
        output_stream.last = (i == NUM_TRANSFERS_OUT - 1);
        output_stream.keep = 0xFFFF;
        output_stream.strb = 0xFFFF;
        out_stream.write(output_stream);
    }
}

void assemble_slider(pixel_t input_data_stored[HEIGHT_IN][WIDTH_IN], pixel_t input_data_slider[SLIDER_HEIGHT_IN][SLIDER_WIDTH_IN], int width_start, int height_start){


}

void disassemble_slider(pixel_t output_data_slider[SLIDER_HEIGHT_OUT][SLIDER_WIDTH_OUT], pixel_t output_data_stored[HEIGHT_OUT][WIDTH_OUT], int width_start, int height_start){


}

// Main function for bilinear interpolation processing
void bilinear_interpolation(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return

	//partition input and output data, still in BRAM -> partitioning array will allow you to write ot multiple mem locations at same time
	//decrease latency
    pixel_t input_data_stored[HEIGHT_IN][WIDTH_IN];
    #pragma HLS BIND_STORAGE variable=input_data_stored type=RAM_2P impl=BRAM
	#pragma HLS array_partition variable=input_data_stored type=complete factor=4

    pixel_t output_data_stored[HEIGHT_OUT][WIDTH_OUT];
    #pragma HLS BIND_STORAGE variable=output_data_stored type=RAM_2P impl=BRAM
	#pragma HLS array_partition variable=output_data_stored type=complete factor=4


    #pragma HLS DATAFLOW
    stream_samples_in(in_stream, input_data_stored);

    for(int i = 0; i < NUM_SLIDERS_HEIGHT; i++){

    	for(int j = 0; j < NUM_SLIDERS_WIDTH; j++){

    		//these should be LUT ram
    		pixel_t temp_data_slider_in[SLIDER_HEIGHT_IN][SLIDER_WIDTH_IN];
			#pragma HLS array_partition variable=temp_data_slider_in type=complete factor=4
    		pixel_t temp_data_slider_out[SLIDER_HEIGHT_OUT][SLIDER_WIDTH_OUT];
			#pragma HLS array_partition variable=temp_data_slider_out type=complete factor=4

    		int height_start_in = i * SLIDER_HEIGHT_IN;
    		int width_start_in = j * SLIDER_WIDTH_IN;
    		int height_start_out = i * SLIDER_HEIGHT_OUT;
    		int width_start_out = j * SLIDER_WIDTH_OUT;

    		//assemble_slider(input_data_stored, temp_data_slider_in, width_start_in, height_start_in);

    		for(int i = 0; i < SLIDER_HEIGHT_IN; i++){
    			for(int j = 0; j < SLIDER_WIDTH_IN; j++){
					#pragma HLS unroll factor=SLIDER_WIDTH_IN
    				temp_data_slider_in[i][j] = input_data_stored[height_start_in + i][width_start_in + j];
    				//int data_to_store = input_data_stored[height_start + i][width_start + j];
    			}
    		}

    		bilinear_interpolation_calculations(temp_data_slider_in, temp_data_slider_out);

    		//try moving this step into bilinear_interpolation_calculations so that once the value is calculated
    		//its stored directly in output_data_stored instead of an output data slider array
    		for(int i = 0; i < SLIDER_HEIGHT_OUT; i++){
    			for(int j = 0; j < SLIDER_WIDTH_OUT; j++){
				#pragma HLS unroll factor=SLIDER_WIDTH_OUT
    				output_data_stored[height_start_out + i][width_start_out + j] = temp_data_slider_out[i][j];
    				//int data_to_store = output_data_slider[i][j];
    			}
    		}

    		//disassemble_slider(temp_data_slider_out, output_data_stored, width_start_out, height_start_out);
    	}
    }

    //bilinear_interpolation_calculations(input_data_stored, output_data_stored);

    stream_samples_out(output_data_stored, out_stream);
}
