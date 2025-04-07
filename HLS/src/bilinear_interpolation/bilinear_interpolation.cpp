#include "bilinear_interpolation.h"
#include <ap_fixed.h>
#include <algorithm>
#include "hls_print.h"


//int bilinear_interpolation_calculations(pixel_t image_section[SLIDER_HEIGHT_IN + BUFFER*2][WIDTH_IN],
//									   int x_start, int y_start,
//									   pixel_t image_section_out[SLIDER_HEIGHT_OUT][SLIDER_WIDTH_OUT]){
int bilinear_interpolation_calculations(pixel_t image_section[SLIDER_HEIGHT_IN + BUFFER*2][WIDTH_IN],
									   int x_start, int y_start,
									   hls::stream<pixel_t> &fifo_bilin_0,
									   hls::stream<pixel_t> &fifo_bilin_1,
									   hls::stream<pixel_t> &fifo_bilin_2,
									   hls::stream<pixel_t> &fifo_bilin_3,
									   hls::stream<pixel_t> &fifo_bilin_4,
									   hls::stream<pixel_t> &fifo_bilin_5,
									   hls::stream<pixel_t> &fifo_bilin_6,
									   hls::stream<pixel_t> &fifo_bilin_7,
									   hls::stream<pixel_t> &fifo_bilin_8,
									   hls::stream<pixel_t> &fifo_bilin_9,
									   hls::stream<pixel_t> &fifo_bilin_10,
									   hls::stream<pixel_t> &fifo_bilin_11,
									   hls::stream<pixel_t> &fifo_bilin_12,
									   hls::stream<pixel_t> &fifo_bilin_13){

	#pragma HLS INTERFACE axis register both port=fifo_bilin_0
	#pragma HLS INTERFACE axis register both port=fifo_bilin_1
	#pragma HLS INTERFACE axis register both port=fifo_bilin_2
	#pragma HLS INTERFACE axis register both port=fifo_bilin_3
	#pragma HLS INTERFACE axis register both port=fifo_bilin_4
	#pragma HLS INTERFACE axis register both port=fifo_bilin_5
	#pragma HLS INTERFACE axis register both port=fifo_bilin_6
	#pragma HLS INTERFACE axis register both port=fifo_bilin_7
	#pragma HLS INTERFACE axis register both port=fifo_bilin_8
	#pragma HLS INTERFACE axis register both port=fifo_bilin_9
	#pragma HLS INTERFACE axis register both port=fifo_bilin_10
	#pragma HLS INTERFACE axis register both port=fifo_bilin_11
	#pragma HLS INTERFACE axis register both port=fifo_bilin_12
	#pragma HLS INTERFACE axis register both port=fifo_bilin_13

	#pragma HLS STREAM variable=fifo_bilin_0 depth=784
	#pragma HLS STREAM variable=fifo_bilin_1 depth=784
	#pragma HLS STREAM variable=fifo_bilin_2 depth=784
	#pragma HLS STREAM variable=fifo_bilin_3 depth=784
	#pragma HLS STREAM variable=fifo_bilin_4 depth=784
	#pragma HLS STREAM variable=fifo_bilin_5 depth=784
	#pragma HLS STREAM variable=fifo_bilin_6 depth=784
	#pragma HLS STREAM variable=fifo_bilin_7 depth=784
	#pragma HLS STREAM variable=fifo_bilin_8 depth=784
	#pragma HLS STREAM variable=fifo_bilin_9 depth=784
	#pragma HLS STREAM variable=fifo_bilin_10 depth=784
	#pragma HLS STREAM variable=fifo_bilin_11 depth=784
	#pragma HLS STREAM variable=fifo_bilin_12 depth=784
	#pragma HLS STREAM variable=fifo_bilin_13 depth=784



	pixel_t image_for_calcs[HEIGHT_IN][WIDTH_IN];
	int section_row = 0;

	//if slice is at top of image
	if(y_start == 0){
		for(int row = 0; row < SLIDER_HEIGHT_IN + 1; row++){

			for (int col = 0; col < WIDTH_IN; col++) {
				#pragma HLS UNROLL
				 image_for_calcs[row][col]= image_section[section_row][col];
			}
			section_row++;
		}

	}

	//if slice is at bottom of image
	else if(y_start == HEIGHT_IN - SLIDER_HEIGHT_IN){
		for(int row = y_start - 1; row < y_start + SLIDER_HEIGHT_IN; row++){

			for (int col = 0; col < WIDTH_IN; col++) {
				#pragma HLS UNROLL
				 image_for_calcs[row][col]= image_section[section_row][col];
			}
			section_row++;
		}
	}

	//if slice is in middle of image
	else if(y_start % SLIDER_HEIGHT_IN == 0){

		for(int row = y_start - 1; row < y_start + SLIDER_HEIGHT_IN + BUFFER; row++){

			for (int col = 0; col < WIDTH_IN; col++) {
				#pragma HLS UNROLL
				 image_for_calcs[row][col]= image_section[section_row][col];
			}
			section_row++;
		}
	}


	int x_location = 0;
	int y_location = 0;

    fixed_t widthRatio  = fixed_t(WIDTH_IN - 1) / fixed_t(WIDTH_OUT - 1);
    fixed_t heightRatio = fixed_t(HEIGHT_IN - 1) / fixed_t(HEIGHT_OUT - 1);

    for (int y_out = y_start * SCALE; y_out < y_start * SCALE + SLIDER_HEIGHT_OUT; ++y_out) {

        #pragma HLS PIPELINE II=11

        for (int x_out = x_start * SCALE; x_out < x_start * SCALE + SLIDER_WIDTH_OUT; ++x_out) {

            #pragma HLS UNROLL factor=2

            // Compute the corresponding input coordinates in fixed point
            fixed_t x_in = x_out * widthRatio;
            fixed_t y_in = y_out * heightRatio;

            // Determine the four nearest neighbors
            int x0 = static_cast<int>(x_in);
            int y0 = static_cast<int>(y_in);
            int x1 = std::min(x0 + 1, WIDTH_IN - 1);
            int y1 = std::min(y0 + 1, HEIGHT_IN - 1);

            // Calculate interpolation weights
            fixed_t dx = x_in - fixed_t(x0);
            fixed_t dy = y_in - fixed_t(y0);

            fixed_t w00 = (fixed_t(1) - dx) * (fixed_t(1) - dy);
            fixed_t w10 = dx * (fixed_t(1) - dy);
            fixed_t w01 = (fixed_t(1) - dx) * dy;
            fixed_t w11 = dx * dy;

            // Read pixel data from input (packed format 0xRRGGBB)
            pixel_t pixel00 = image_for_calcs[y0][x0];
            pixel_t pixel10 = image_for_calcs[y0][x1];
            pixel_t pixel01 = image_for_calcs[y1][x0];
            pixel_t pixel11 = image_for_calcs[y1][x1];

            // Extract RGB channels
            channel_t b00 = (pixel00 >> 16) & 0xFF, g00 = (pixel00 >> 8) & 0xFF, r00 = pixel00 & 0xFF;
            channel_t b10 = (pixel10 >> 16) & 0xFF, g10 = (pixel10 >> 8) & 0xFF, r10 = pixel10 & 0xFF;
            channel_t b01 = (pixel01 >> 16) & 0xFF, g01 = (pixel01 >> 8) & 0xFF, r01 = pixel01 & 0xFF;
            channel_t b11 = (pixel11 >> 16) & 0xFF, g11 = (pixel11 >> 8) & 0xFF, r11 = pixel11 & 0xFF;

            // Compute interpolated values for each channel
            fixed_t r_interp = w00 * fixed_t(r00) + w10 * fixed_t(r10) + w01 * fixed_t(r01) + w11 * fixed_t(r11);
            fixed_t g_interp = w00 * fixed_t(g00) + w10 * fixed_t(g10) + w01 * fixed_t(g01) + w11 * fixed_t(g11);
            fixed_t b_interp = w00 * fixed_t(b00) + w10 * fixed_t(b10) + w01 * fixed_t(b01) + w11 * fixed_t(b11);

            // Store interpolated values in `image_out` (rounded)
            pixel_t temp_pixel;
            temp_pixel.range(7, 0) = r_interp;
            temp_pixel.range(15, 8) = g_interp;
            temp_pixel.range(23, 16) = b_interp;

            //image_section_out[y_location][x_location] = temp_pixel;
            switch (y_location) {
                case 0:  fifo_bilin_0.write(temp_pixel);  break;
                case 1:  fifo_bilin_1.write(temp_pixel);  break;
                case 2:  fifo_bilin_2.write(temp_pixel);  break;
                case 3:  fifo_bilin_3.write(temp_pixel);  break;
                case 4:  fifo_bilin_4.write(temp_pixel);  break;
                case 5:  fifo_bilin_5.write(temp_pixel);  break;
                case 6:  fifo_bilin_6.write(temp_pixel);  break;
                case 7:  fifo_bilin_7.write(temp_pixel);  break;
                case 8:  fifo_bilin_8.write(temp_pixel);  break;
                case 9:  fifo_bilin_9.write(temp_pixel);  break;
                case 10: fifo_bilin_10.write(temp_pixel); break;
                case 11: fifo_bilin_11.write(temp_pixel); break;
                case 12: fifo_bilin_12.write(temp_pixel); break;
                case 13: fifo_bilin_13.write(temp_pixel); break;
            }

            x_location++;
        }

        x_location = 0;
        y_location++;
    }

    return 1;
}


void stream_samples_in(hls::stream<axis_t> &in_stream,
                       hls::stream<pixel_t> &fifo_first_0,
                       hls::stream<pixel_t> &fifo_first_1,
                       hls::stream<pixel_t> &fifo_first_2,
                       hls::stream<pixel_t> &fifo_first_3,
                       hls::stream<pixel_t> &fifo_first_4,
                       hls::stream<pixel_t> &fifo_first_5,
                       hls::stream<pixel_t> &fifo_first_6,
                       hls::stream<pixel_t> &fifo_first_7,
                       hls::stream<pixel_t> &fifo_first_8,
                       hls::stream<pixel_t> &fifo_second_0,
                       hls::stream<pixel_t> &fifo_second_1,
                       hls::stream<pixel_t> &fifo_second_2,
                       hls::stream<pixel_t> &fifo_second_3,
                       hls::stream<pixel_t> &fifo_second_4,
                       hls::stream<pixel_t> &fifo_second_5,
                       hls::stream<pixel_t> &fifo_second_6,
                       hls::stream<pixel_t> &fifo_second_7,
                       hls::stream<pixel_t> &fifo_second_8){

    #pragma HLS INTERFACE axis register both port=in_stream
    #pragma HLS INTERFACE axis register both port=fifo_first_0
//#pragma HLS RESOURCE variable=psum1 core=FIFO_BRAM
    #pragma HLS INTERFACE axis register both port=fifo_first_1
    #pragma HLS INTERFACE axis register both port=fifo_first_2
    #pragma HLS INTERFACE axis register both port=fifo_first_3
    #pragma HLS INTERFACE axis register both port=fifo_first_4
    #pragma HLS INTERFACE axis register both port=fifo_first_5
    #pragma HLS INTERFACE axis register both port=fifo_first_6
    #pragma HLS INTERFACE axis register both port=fifo_first_7
    #pragma HLS INTERFACE axis register both port=fifo_first_8
    #pragma HLS INTERFACE axis register both port=fifo_second_0
    #pragma HLS INTERFACE axis register both port=fifo_second_1
    #pragma HLS INTERFACE axis register both port=fifo_second_2
    #pragma HLS INTERFACE axis register both port=fifo_second_3
    #pragma HLS INTERFACE axis register both port=fifo_second_4
    #pragma HLS INTERFACE axis register both port=fifo_second_5
    #pragma HLS INTERFACE axis register both port=fifo_second_6
    #pragma HLS INTERFACE axis register both port=fifo_second_7
    #pragma HLS INTERFACE axis register both port=fifo_second_8

	#pragma HLS STREAM variable=fifo_first_0 depth=784
	#pragma HLS STREAM variable=fifo_first_1 depth=784
	#pragma HLS STREAM variable=fifo_first_2 depth=784
	#pragma HLS STREAM variable=fifo_first_3 depth=784
	#pragma HLS STREAM variable=fifo_first_4 depth=784
	#pragma HLS STREAM variable=fifo_first_5 depth=784
	#pragma HLS STREAM variable=fifo_first_6 depth=784
	#pragma HLS STREAM variable=fifo_first_7 depth=784
	#pragma HLS STREAM variable=fifo_first_8 depth=784
	#pragma HLS STREAM variable=fifo_second_0 depth=784
	#pragma HLS STREAM variable=fifo_second_1 depth=784
	#pragma HLS STREAM variable=fifo_second_2 depth=784
	#pragma HLS STREAM variable=fifo_second_3 depth=784
	#pragma HLS STREAM variable=fifo_second_4 depth=784
	#pragma HLS STREAM variable=fifo_second_5 depth=784
	#pragma HLS STREAM variable=fifo_second_6 depth=784
	#pragma HLS STREAM variable=fifo_second_7 depth=784
	#pragma HLS STREAM variable=fifo_second_8 depth=784


    // Internal overlap FIFOs
    hls::stream<pixel_t> fifo_first_overlap_0 ("overlap 1 0");
    hls::stream<pixel_t> fifo_first_overlap_1 ("overlap 1 1");
    hls::stream<pixel_t> fifo_second_overlap_0 ("overlap 2 0");
    hls::stream<pixel_t> fifo_second_overlap_1 ("overlap 2 1");

	#pragma HLS INTERFACE axis register both port=fifo_first_overlap_0
	#pragma HLS INTERFACE axis register both port=fifo_first_overlap_1
	#pragma HLS INTERFACE axis register both port=fifo_second_overlap_0
	#pragma HLS INTERFACE axis register both port=fifo_second_overlap_1

	#pragma HLS STREAM variable=fifo_first_overlap_0 depth=56
	#pragma HLS STREAM variable=fifo_first_overlap_1 depth=56
	#pragma HLS STREAM variable=fifo_second_overlap_0 depth=56
	#pragma HLS STREAM variable=fifo_second_overlap_1 depth=56


    /*STREAM IN VALUES*/
    data_streamed input_streams_stored[NUM_TRANSFERS_IN];

    //Read in the streams and store them
    int i = 0;
	while (i < NUM_TRANSFERS_IN) {
		while (!in_stream.empty()) {

			if (i == NUM_TRANSFERS_IN){
				break;
			}

			axis_t temp_input = in_stream.read();
			input_streams_stored[i] = temp_input.data;
			i++;
		}
	}


	/*PARSE VALUES INTO PIXELS*/
	pixel_t input_data_stored[HEIGHT_IN][WIDTH_IN];

	for (int transfer = 0; transfer < NUM_TRANSFERS_IN; transfer++) {

		//splits transfer into individual pixels and stores them in a data array
		for (int j = 0; j < PIXELS_PER_TRANSFER; j++) {

			int index = transfer * PIXELS_PER_TRANSFER + j;
			int y = index / WIDTH_IN;
			int x = index % WIDTH_IN;
			//int input_data_stored_value = input_streams_stored[transfer].range(j * BITS_PER_PIXEL + 23, j * BITS_PER_PIXEL);
			input_data_stored[y][x] = input_streams_stored[transfer].range(j * BITS_PER_PIXEL + 23, j * BITS_PER_PIXEL);
		}
	}


	/*STORE PIXEL VALUES IN FIFOS*/
	int fifo_idx = 0;
	bool fill_first_set = true;

	//every row will be stored in a FIFO
	for(int row_idx = 0; row_idx < HEIGHT_IN; row_idx++){

		//determine which fifo is being written into next
		if(row_idx < SLIDER_HEIGHT_IN + BUFFER){
			fifo_idx = row_idx + 1;
		}
		else{
			fifo_idx = (row_idx - (SLIDER_HEIGHT_IN + BUFFER)) % SLIDER_HEIGHT_IN + BUFFER * 2;
		}


		//fill FIFOs with pixel values
		for(int col_idx = 0; col_idx < WIDTH_IN; col_idx++){
			pixel_t data = input_data_stored[row_idx][col_idx];

			switch (fifo_idx) {
			    case 0:
			        if (fill_first_set) fifo_first_0.write(data);
			        else fifo_second_0.write(data);
			        break;
			    case 1:
			        if (fill_first_set) fifo_first_1.write(data);
			        else fifo_second_1.write(data);
			        break;
			    case 2:
			        if (fill_first_set) fifo_first_2.write(data);
			        else fifo_second_2.write(data);
			        break;
			    case 3:
			        if (fill_first_set) fifo_first_3.write(data);
			        else fifo_second_3.write(data);
			        break;
			    case 4:
			        if (fill_first_set) fifo_first_4.write(data);
			        else fifo_second_4.write(data);
			        break;
			    case 5:
			        if (fill_first_set) fifo_first_5.write(data);
			        else fifo_second_5.write(data);
			        break;
			    case 6:
			        if (fill_first_set) fifo_first_6.write(data);
			        else fifo_second_6.write(data);
			        break;
			    case 7:
			        if (fill_first_set) fifo_first_7.write(data);
			        else fifo_second_7.write(data);
			        break;
			    case 8:
			        if (fill_first_set) fifo_first_8.write(data);
			        else fifo_second_8.write(data);
			        break;
			}



			//if not at bottom of image
			if(row_idx != HEIGHT_IN - 1){
				//if if at the bottom two rows of the top image section, fill first FIFO overflow buffer
				if(row_idx == SLIDER_HEIGHT_IN - 1){
					fifo_first_overlap_0.write(data);
				}
				else if(row_idx == SLIDER_HEIGHT_IN){
					fifo_first_overlap_1.write(data);
				}


				//otherwise if FIFOs at the bottom of the middle image sections are being filled, alternate which FIFO overflow buffer is filled
				else if(fifo_idx == SLIDER_HEIGHT_IN){
					if(fill_first_set){
						fifo_first_overlap_0.write(data);
					}
					else if(!fill_first_set){
						fifo_second_overlap_0.write(data);
					}
				}
				else if(fifo_idx == (SLIDER_HEIGHT_IN + 1)){
					if(fill_first_set){
						fifo_first_overlap_1.write(data);
					}
					else if(!fill_first_set){
						fifo_second_overlap_1.write(data);
					}
				}
			}



		}


		/*SWITCH WHICH FIFOS BEING FILLED AND TRANSFER OVERLAP*/
		if(fifo_idx == (SLIDER_HEIGHT_IN + 1)){
			fill_first_set = !fill_first_set;

			for(int col_idx = 0; col_idx < WIDTH_IN; col_idx++){
				//NOTE: Add checks to confirm overlap has values and handle if they don't
				if(fill_first_set && !fifo_second_overlap_0.empty() && !fifo_second_overlap_1.empty()){

					fifo_first_0.write(fifo_second_overlap_0.read());
					fifo_first_1.write(fifo_second_overlap_1.read());

				}
				else if(!fill_first_set && !fifo_first_overlap_0.empty() && !fifo_first_overlap_1.empty()){


					fifo_second_0.write(fifo_first_overlap_0.read());
					fifo_second_1.write(fifo_first_overlap_1.read());

				}
			}


		}



	}

}

void create_image_section(hls::stream<pixel_t> &fifo_0,
                          hls::stream<pixel_t> &fifo_1,
                          hls::stream<pixel_t> &fifo_2,
                          hls::stream<pixel_t> &fifo_3,
                          hls::stream<pixel_t> &fifo_4,
                          hls::stream<pixel_t> &fifo_5,
                          hls::stream<pixel_t> &fifo_6,
                          hls::stream<pixel_t> &fifo_7,
                          hls::stream<pixel_t> &fifo_8,
                          bool top_slider, bool bottom_slider,
                          pixel_t image_section[SLIDER_HEIGHT_IN + BUFFER * 2][WIDTH_IN]) {

    // If slider is in top section, fill from FIFOs 1-8
    if (top_slider) {
        for (int col_idx = 0; col_idx < WIDTH_IN; col_idx++) {
            for (int i = 0; i < SLIDER_HEIGHT_IN + BUFFER; i++) {
				pixel_t data = (i == 0) ? fifo_1.read() :
							   (i == 1) ? fifo_2.read() :
							   (i == 2) ? fifo_3.read() :
							   (i == 3) ? fifo_4.read() :
							   (i == 4) ? fifo_5.read() :
							   (i == 5) ? fifo_6.read() :
							   (i == 6) ? fifo_7.read() :
							   fifo_8.read();
				image_section[i][col_idx] = data;
            }
        }
    }
    // If slider is in bottom section, fill from FIFOs 0-7
    else if (bottom_slider) {
        for (int col_idx = 0; col_idx < WIDTH_IN; col_idx++) {
            for (int i = 0; i < SLIDER_HEIGHT_IN + BUFFER; i++) {
				pixel_t data = (i == 0) ? fifo_0.read() :
							   (i == 1) ? fifo_1.read() :
							   (i == 2) ? fifo_2.read() :
							   (i == 3) ? fifo_3.read() :
							   (i == 4) ? fifo_4.read() :
							   (i == 5) ? fifo_5.read() :
							   (i == 6) ? fifo_6.read() :
							   fifo_7.read();
				image_section[i][col_idx] = data;
            }
        }
    }
    // If slider is not on top or bottom, fill from FIFOs 0-8
    else {
        for (int col_idx = 0; col_idx < WIDTH_IN; col_idx++) {
            for (int i = 0; i < SLIDER_HEIGHT_IN + BUFFER * 2; i++) {
                pixel_t data = (i == 0) ? fifo_0.read() :
                               (i == 1) ? fifo_1.read() :
                               (i == 2) ? fifo_2.read() :
                               (i == 3) ? fifo_3.read() :
                               (i == 4) ? fifo_4.read() :
                               (i == 5) ? fifo_5.read() :
                               (i == 6) ? fifo_6.read() :
                               (i == 7) ? fifo_7.read() :
                               fifo_8.read();
                image_section[i][col_idx] = data;
            }
        }
    }
}



void create_upscaled_image_section(hls::stream<pixel_t> &fifo_0,
									hls::stream<pixel_t> &fifo_1,
									hls::stream<pixel_t> &fifo_2,
									hls::stream<pixel_t> &fifo_3,
									hls::stream<pixel_t> &fifo_4,
									hls::stream<pixel_t> &fifo_5,
									hls::stream<pixel_t> &fifo_6,
									hls::stream<pixel_t> &fifo_7,
									hls::stream<pixel_t> &fifo_8,
									hls::stream<pixel_t> &fifo_9,
									hls::stream<pixel_t> &fifo_10,
									hls::stream<pixel_t> &fifo_11,
									hls::stream<pixel_t> &fifo_12,
									hls::stream<pixel_t> &fifo_13,
									pixel_t image_section[SLIDER_HEIGHT_OUT][SLIDER_WIDTH_OUT]){

    for (int col_idx = 0; col_idx < SLIDER_WIDTH_OUT; col_idx++) {
    	image_section[0][col_idx] = fifo_0.read();
    	image_section[1][col_idx] = fifo_1.read();
    	image_section[2][col_idx] = fifo_2.read();
    	image_section[3][col_idx] = fifo_3.read();
    	image_section[4][col_idx] = fifo_4.read();
    	image_section[5][col_idx] = fifo_5.read();
    	image_section[6][col_idx] = fifo_6.read();
    	image_section[7][col_idx] = fifo_7.read();
    	image_section[8][col_idx] = fifo_8.read();
    	image_section[9][col_idx] = fifo_9.read();
    	image_section[10][col_idx] = fifo_10.read();
    	image_section[11][col_idx] = fifo_11.read();
    	image_section[12][col_idx] = fifo_12.read();
    	image_section[13][col_idx] = fifo_13.read();
    }

}



void combine_upscaled_sections(pixel_t upscaled_sections[NUM_SLIDERS][SLIDER_HEIGHT_OUT][SLIDER_WIDTH_OUT],
							   pixel_t output_data_stored[HEIGHT_OUT][WIDTH_OUT]) {

    //#pragma HLS ARRAY_PARTITION variable=upscaled_sections complete dim=1
    //#pragma HLS ARRAY_PARTITION variable=output_data_stored complete dim=1

    // Iterate over the grid of slaces
    for (int row_group = 0; row_group < NUM_SLIDERS_HEIGHT; row_group++) {
        for (int col_group = 0; col_group < NUM_SLIDERS_WIDTH; col_group++) {
            // Compute the starting positions in the output array
            int start_row = row_group * SLIDER_HEIGHT_OUT;
            int start_col = col_group * SLIDER_WIDTH_OUT;
            int slice_idx = row_group * NUM_SLIDERS_WIDTH + col_group;

            // Copy slice data into the final combined array
            for (int i = 0; i < SLIDER_HEIGHT_OUT; i++) {
                for (int j = 0; j < SLIDER_WIDTH_OUT; j++) {
                    #pragma HLS PIPELINE
                    output_data_stored[start_row + i][start_col + j] = upscaled_sections[slice_idx][i][j];
                }
            }
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


// Main function for bilinear interpolation processing
void bilinear_interpolation(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return

	// Declare FIFOs for pixel values
	hls::stream<pixel_t> fifo_first_0 ("first 0"), fifo_first_1 ("first 1"), fifo_first_2, fifo_first_3, fifo_first_4, fifo_first_5, fifo_first_6, fifo_first_7, fifo_first_8;
	hls::stream<pixel_t> fifo_second_0, fifo_second_1, fifo_second_2, fifo_second_3, fifo_second_4, fifo_second_5, fifo_second_6, fifo_second_7, fifo_second_8;

	hls::stream<pixel_t> fifo_bilin_0, fifo_bilin_1, fifo_bilin_2, fifo_bilin_3,
	                      fifo_bilin_4, fifo_bilin_5, fifo_bilin_6, fifo_bilin_7,
	                      fifo_bilin_8, fifo_bilin_9, fifo_bilin_10, fifo_bilin_11,
	                      fifo_bilin_12, fifo_bilin_13;

	//Stream values in and store pixel values in FIFOs
#pragma HLS DATAFLOW
	stream_samples_in(in_stream,
	                  fifo_first_0, fifo_first_1, fifo_first_2, fifo_first_3, fifo_first_4, fifo_first_5, fifo_first_6, fifo_first_7, fifo_first_8,
	                  fifo_second_0, fifo_second_1, fifo_second_2, fifo_second_3, fifo_second_4, fifo_second_5, fifo_second_6, fifo_second_7, fifo_second_8);



	//Create image subsections from FIFO values
	bool first_fifos_filled = false, second_fifos_filled = false;

	//combine FIFOs into slices and upscale slices
	int row_sliced = 0;
	pixel_t image_section[SLIDER_HEIGHT_IN + BUFFER * 2][WIDTH_IN];

	pixel_t upscaled_sections[NUM_SLIDERS][SLIDER_HEIGHT_OUT][SLIDER_WIDTH_OUT];
	int section_upscaled = 0;

	while(row_sliced < NUM_SLIDERS_HEIGHT){

		//if the second set of FIFOs has data then the first set is filled
		if(!fifo_second_0.empty()){
			first_fifos_filled = true;
		}

		//won't be immediately triggered because first fifo filled is fifo_first_1 not fifo_first_0
		if(!fifo_first_0.empty()){
			second_fifos_filled = true;
		}

		bool top_row = (row_sliced == 0);
		bool bottom_row = (row_sliced == NUM_SLIDERS_HEIGHT - 1);

		//if row_sliced number is even then pulling from second set of FIFOs
		if(row_sliced % 2 == 0 && first_fifos_filled){

#pragma HLS DATAFLOW
			create_image_section(fifo_first_0, fifo_first_1, fifo_first_2, fifo_first_3, fifo_first_4, fifo_first_5, fifo_first_6, fifo_first_7, fifo_first_8,
								top_row, bottom_row, image_section);
			first_fifos_filled = false;
			//CALL BILINEAR INTERPOLATION -> # of times called = NUM_SLIDERS_WIDTH
			for (int i = 0; i < NUM_SLIDERS_WIDTH; i++) {

				bilinear_interpolation_calculations(
				    image_section,
				    SLIDER_WIDTH_IN * i,
				    row_sliced * SLIDER_HEIGHT_IN,
				    fifo_bilin_0, fifo_bilin_1, fifo_bilin_2, fifo_bilin_3,
				    fifo_bilin_4, fifo_bilin_5, fifo_bilin_6, fifo_bilin_7,
				    fifo_bilin_8, fifo_bilin_9, fifo_bilin_10, fifo_bilin_11,
				    fifo_bilin_12, fifo_bilin_13);

				create_upscaled_image_section(
						fifo_bilin_0, fifo_bilin_1, fifo_bilin_2, fifo_bilin_3,
						fifo_bilin_4, fifo_bilin_5, fifo_bilin_6, fifo_bilin_7,
						fifo_bilin_8, fifo_bilin_9, fifo_bilin_10, fifo_bilin_11,
						fifo_bilin_12, fifo_bilin_13, upscaled_sections[section_upscaled + i]);

			}
			section_upscaled = section_upscaled + NUM_SLIDERS_WIDTH;
			row_sliced++;
		}

		//if the row_sliced number is odd, then pulling from second set of FIFOs
		else if(row_sliced % 2 == 1 && second_fifos_filled){
#pragma HLS DATAFLOW
			create_image_section(fifo_second_0, fifo_second_1, fifo_second_2, fifo_second_3, fifo_second_4, fifo_second_5, fifo_second_6, fifo_second_7, fifo_second_8,
								top_row, bottom_row, image_section);
			second_fifos_filled = false;
			//CALL BILINEAR INTERPOLATION
			for (int i = 0; i < NUM_SLIDERS_WIDTH; i++) {

				bilinear_interpolation_calculations(
				    image_section,
				    SLIDER_WIDTH_IN * i,
				    row_sliced * SLIDER_HEIGHT_IN,
				    fifo_bilin_0, fifo_bilin_1, fifo_bilin_2, fifo_bilin_3,
				    fifo_bilin_4, fifo_bilin_5, fifo_bilin_6, fifo_bilin_7,
				    fifo_bilin_8, fifo_bilin_9, fifo_bilin_10, fifo_bilin_11,
				    fifo_bilin_12, fifo_bilin_13);

				create_upscaled_image_section(
						fifo_bilin_0, fifo_bilin_1, fifo_bilin_2, fifo_bilin_3,
						fifo_bilin_4, fifo_bilin_5, fifo_bilin_6, fifo_bilin_7,
						fifo_bilin_8, fifo_bilin_9, fifo_bilin_10, fifo_bilin_11,
						fifo_bilin_12, fifo_bilin_13, upscaled_sections[section_upscaled + i]);


			}
			section_upscaled = section_upscaled + NUM_SLIDERS_WIDTH;
			row_sliced++;
		}
	}


    pixel_t output_data_stored[HEIGHT_OUT][WIDTH_OUT];
    #pragma HLS BIND_STORAGE variable=output_data_stored type=RAM_2P impl=BRAM

    combine_upscaled_sections(upscaled_sections, output_data_stored);

    stream_samples_out(output_data_stored, out_stream);
}
