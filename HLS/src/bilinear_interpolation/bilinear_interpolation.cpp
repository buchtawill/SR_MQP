#include "bilinear_interpolation.h"
#include <ap_fixed.h>
#include <algorithm>

// HLS function for bilinear interpolation
int bilinear_interpolation_calculations(pixel_t image_in[SLIDER_HEIGHT_IN][SLIDER_WIDTH_IN],
                                        pixel_t image_out[SLIDER_HEIGHT_OUT][SLIDER_WIDTH_OUT]) {


    fixed widthRatio  = fixed(SLIDER_WIDTH_WO_BUFFER - 1) / fixed(SLIDER_WIDTH_OUT - 1);
    fixed heightRatio = fixed(SLIDER_HEIGHT_WO_BUFFER - 1) / fixed(SLIDER_HEIGHT_OUT - 1);

    for (int y_out = 0; y_out < SLIDER_HEIGHT_OUT; ++y_out) {

        #pragma HLS PIPELINE II=11

        for (int x_out = 0; x_out < SLIDER_WIDTH_OUT; ++x_out) {

            #pragma HLS UNROLL factor=2

            // Compute the corresponding input coordinates in fixed point
            fixed x_in = x_out * widthRatio + 1;  // Offset by 1 to account for buffer
            fixed y_in = y_out * heightRatio + 1; // Offset by 1 to account for buffer

            // Determine the four nearest neighbors
            int x0 = static_cast<int>(x_in);
            int y0 = static_cast<int>(y_in);
            int x1 = std::min(x0 + 1, SLIDER_WIDTH_WO_BUFFER - 1);
            int y1 = std::min(y0 + 1, SLIDER_HEIGHT_WO_BUFFER - 1);

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
//void stream_samples_in(hls::stream<axis_t> &in_stream, pixel_t input_data_stored[HEIGHT_IN][WIDTH_IN]) {
void stream_samples_in(hls::stream<axis_t> &in_stream, hls::stream<data_streamed, PIXELS_PER_TRANSFER> &data_fifo_in){

    int i = 0;

    //data_streamed input_streams_stored[NUM_TRANSFERS];

    while (i < NUM_TRANSFERS) {

        while (!in_stream.empty()) {

            if (i == NUM_TRANSFERS){
            	break;
            }

            axis_t temp_input = in_stream.read();
            data_streamed temp_data = temp_input.data;
            data_fifo_in.write(temp_data);

            i++;
        }
    }

}

void fill_image_fifos(hls::stream<data_streamed, PIXELS_PER_TRANSFER> &input_stream,
					  hls::stream<pixel_t, WIDTH_IN> &storage_fifo_overlap_top,
					  hls::stream<pixel_t, WIDTH_IN> &storage_fifo_0,
					  hls::stream<pixel_t, WIDTH_IN> &storage_fifo_1,
					  hls::stream<pixel_t, WIDTH_IN> &storage_fifo_2,
					  hls::stream<pixel_t, WIDTH_IN> &storage_fifo_3,
					  hls::stream<pixel_t, WIDTH_IN> &storage_fifo_4,
					  hls::stream<pixel_t, WIDTH_IN> &storage_fifo_overlap_bottom) {


    int count = 0;  // Keeps track of how many values have been written to the current FIFO
    int fifo_index = 0;  // Tracks which FIFO is currently being filled
    bool first_fill = true;

    while (!input_stream.empty()) {

        data_streamed temp_data = input_stream.read();  // Read from input stream
        pixel_t data_split[PIXELS_PER_TRANSFER];

        //Handle filling the top overlap stream either with 0s or values used to be on the bottom
        if (fifo_index == 0) {

        	for(int i = 0; i < PIXELS_PER_TRANSFER; i++){
				if (first_fill) {
					// Fill storage_fifo_overlap_top with 0s on the first fill of fifo_0
					storage_fifo_overlap_top.write((pixel_t)0);

					//if top row is fully filled with zeros have next write fill from bottom buffer
					if(i == PIXELS_PER_TRANSFER - 1){
						first_fill = false;
					}

				} else {
					// Fill storage_fifo_overlap_top with values from storage_fifo_overlap_bottom
					if (!storage_fifo_overlap_bottom.empty()) {
						storage_fifo_overlap_top.write(storage_fifo_overlap_bottom.read());
					}
				}
        	}
        }

        //split input stream into pixels
        for (int pixel_transfer = 0; pixel_transfer < PIXELS_PER_TRANSFER; pixel_transfer++) {
        	data_split[pixel_transfer] = temp_data.range(pixel_transfer * BITS_PER_PIXEL + 23, pixel_transfer * BITS_PER_PIXEL);
        }


        // Write data to the appropriate FIFO
        switch (fifo_index) {
            case 0:

                if (first_fill) {
                    // Fill storage_fifo_overlap_top with 0s on the first fill of fifo_0
                    storage_fifo_overlap_top.write((pixel_t)0);
                    first_fill = false;
                } else {
                    // Fill storage_fifo_overlap_top with values from storage_fifo_overlap_bottom
                    if (!storage_fifo_overlap_bottom.empty()) {
                        storage_fifo_overlap_top.write(storage_fifo_overlap_bottom.read());
                    }
                }

                for(int i = 0; i < PIXELS_PER_TRANSFER; i++){
                	storage_fifo_0.write(data_split[i]);
                }

            	break;
            case 1:

                for(int i = 0; i < PIXELS_PER_TRANSFER; i++){
                	storage_fifo_1.write(data_split[i]);
                }

            	break;
            case 2:

                for(int i = 0; i < PIXELS_PER_TRANSFER; i++){
                	storage_fifo_2.write(data_split[i]);
                }

            	break;
            case 3:

                for(int i = 0; i < PIXELS_PER_TRANSFER; i++){
                	storage_fifo_3.write(data_split[i]);
                }

            	break;
            case 4:

                for(int i = 0; i < PIXELS_PER_TRANSFER; i++){
                	storage_fifo_4.write(data_split[i]);
                	storage_fifo_overlap_bottom.write(data_split[i]);
                }

                break;
        }

        count++;

        // Switch to the next FIFO after 7 writes
        if (count == 7) {
            count = 0;
            fifo_index = (fifo_index + 1) % 5;  // Cycle back to FIFO 0 after FIFO 4
        }
    }
}


// Function to stream output samples from 2D array
//void stream_samples_out(pixel_t output_data_stored[HEIGHT_OUT][WIDTH_OUT], hls::stream<axis_t> &out_stream) {
void stream_samples_out(hls::stream<data_streamed, PIXELS_PER_TRANSFER> &data_fifo_out, hls::stream<axis_t> &out_stream){
    //data_streamed loaded[NUM_TRANSFERS_OUT];

    for (int i = 0; i < NUM_TRANSFERS_OUT; i++) {
        axis_t output_stream;
        output_stream.data = data_fifo_out.read();
        output_stream.last = (i == NUM_TRANSFERS_OUT - 1);
        output_stream.keep = 0xFFFF;
        output_stream.strb = 0xFFFF;
        out_stream.write(output_stream);
    }
}

void fill_output_fifo(pixel_t output_data_stored[HEIGHT_OUT][WIDTH_OUT], hls::stream<data_streamed, PIXELS_PER_TRANSFER> &output_fifo){

	for (int load = 0; load < NUM_TRANSFERS_OUT; load++) {

        data_streamed temp_load = 0;

        for (int pixel_transfer = 0; pixel_transfer < PIXELS_PER_TRANSFER; pixel_transfer++) {
            int index = load * PIXELS_PER_TRANSFER + pixel_transfer;
            int y = index / WIDTH_OUT;
            int x = index % WIDTH_OUT;

            temp_load.range(pixel_transfer * BITS_PER_PIXEL + 23, pixel_transfer * BITS_PER_PIXEL) = output_data_stored[y][x];
        }

        output_fifo.write(temp_load);
    }
}




// Main function for bilinear interpolation processing
void bilinear_interpolation(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return

	hls::stream<data_streamed, PIXELS_PER_TRANSFER> input_fifo;
	//#pragma HLS BIND_STORAGE variable=input_fifo  type=ram_2p impl=bram
	hls::stream<data_streamed, PIXELS_PER_TRANSFER> output_fifo;
	//#pragma HLS BIND_STORAGE variable=output_fifo type=ram_2p impl=bram

	hls::stream<pixel_t, WIDTH_IN> storage_fifo_0;
	hls::stream<pixel_t, WIDTH_IN> storage_fifo_1;
	hls::stream<pixel_t, WIDTH_IN> storage_fifo_2;
	hls::stream<pixel_t, WIDTH_IN> storage_fifo_3;
	hls::stream<pixel_t, WIDTH_IN> storage_fifo_4;

	hls::stream<pixel_t, WIDTH_IN> storage_fifo_overlap_top;
	hls::stream<pixel_t, WIDTH_IN> storage_fifo_overlap_bottom;


    pixel_t output_data_stored[HEIGHT_OUT][WIDTH_OUT];
    //#pragma HLS BIND_STORAGE variable=output_data_stored type=RAM_2P impl=BRAM


    //#pragma HLS DATAFLOW
    stream_samples_in(in_stream, input_fifo);
    fill_image_fifos(input_fifo, storage_fifo_overlap_top,
    				 storage_fifo_0, storage_fifo_1, storage_fifo_2,
    				 storage_fifo_3, storage_fifo_4, storage_fifo_overlap_bottom);


    //local caches for storing values on edges used more than once
    pixel_t edge_row_0 = 0, edge_row_1 = 0, edge_row_2 = 0, edge_row_3 = 0, edge_row_4 = 0, edge_row_5 = 0;


    int i = 0, j = 0;

    //SPLIT IMAGE INTO SLIDING ARRAY
    while(i < NUM_SLIDERS_HEIGHT){

		if(!storage_fifo_0.empty() && !storage_fifo_1.empty() && !storage_fifo_2.empty() &&
		   !storage_fifo_3.empty() && !storage_fifo_4.empty()){

			while(j < NUM_SLIDERS_WIDTH){

				//SWITCH OVER TO FIFOS

				//these should be LUT ram
				pixel_t temp_data_slider_in[SLIDER_HEIGHT_IN][SLIDER_WIDTH_IN];
				//#pragma HLS array_partition variable=temp_data_slider_in type=complete factor=4
				pixel_t temp_data_slider_out[SLIDER_HEIGHT_OUT][SLIDER_WIDTH_OUT];
				//#pragma HLS array_partition variable=temp_data_slider_out type=complete factor=4



				//will need to manually update if we want to change sliders height
				//handles overlap on left and right
				for(int col = 0; col < SLIDER_WIDTH_IN; col++){

					//handle image overlap to the left
					if(col == 0){
						temp_data_slider_in[0][col] = edge_row_0;
						temp_data_slider_in[1][col] = edge_row_1;
						temp_data_slider_in[2][col] = edge_row_2;
						temp_data_slider_in[3][col] = edge_row_3;
						temp_data_slider_in[4][col] = edge_row_4;
						temp_data_slider_in[5][col] = edge_row_5;
					}
					else{
						pixel_t temp_0 = storage_fifo_overlap_top.read();
						pixel_t temp_1 = storage_fifo_0.read();
						pixel_t temp_2 = storage_fifo_1.read();
						pixel_t temp_3 = storage_fifo_2.read();
						pixel_t temp_4 = storage_fifo_3.read();
						pixel_t temp_5 = storage_fifo_4.read();

						//if not in the overlap on the right
						if(col < SLIDER_WIDTH_IN - 2){
							temp_data_slider_in[0][col] = temp_0;
							temp_data_slider_in[1][col] = temp_1;
							temp_data_slider_in[2][col] = temp_2;
							temp_data_slider_in[3][col] = temp_3;
							temp_data_slider_in[4][col] = temp_4;
							temp_data_slider_in[5][col] = temp_5;
						}
						//handle image overlap on the right
						else{
							temp_data_slider_in[0][col] = temp_0;
							temp_data_slider_in[1][col] = temp_1;
							temp_data_slider_in[2][col] = temp_2;
							temp_data_slider_in[3][col] = temp_3;
							temp_data_slider_in[4][col] = temp_4;
							temp_data_slider_in[5][col] = temp_5;

							edge_row_0 = temp_0;
							edge_row_1 = temp_1;
							edge_row_2 = temp_2;
							edge_row_3 = temp_3;
							edge_row_4 = temp_4;
							edge_row_5 = temp_5;
						}
					}
				}

				bilinear_interpolation_calculations(temp_data_slider_in, temp_data_slider_out);

				int height_start_out = i * SLIDER_HEIGHT_OUT;
				int width_start_out = j * SLIDER_WIDTH_OUT;

				//try moving this step into bilinear_interpolation_calculations so that once the value is calculated
				//its stored directly in output_data_stored instead of an output data slider array
				for(int row = 0; row < SLIDER_HEIGHT_OUT; row++){
					for(int col = 0; col < SLIDER_WIDTH_OUT; col++){
					//#pragma HLS unroll factor=SLIDER_WIDTH_OUT
						output_data_stored[height_start_out + row][width_start_out + col] = temp_data_slider_out[row][col];
					}
				}

				j++;

			}

			i++;
		}
    }

    fill_output_fifo(output_data_stored, output_fifo);
    stream_samples_out(output_fifo, out_stream);
}
