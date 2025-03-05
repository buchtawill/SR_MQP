#include "conv2d.h"
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <iostream>
#include <stdio.h>
#include "fsrcnn_weights.h"

void fill_input_fifo(hls::stream<axis_t> &axis_in,
					 hls::stream<stream_data_t, STREAM_BEATS_PER_TILE> &data_fifo_in){

	axis_t tmp_stream;
	tmp_stream.last = 0;

	int in_ptr_abs = 0;
	while(!tmp_stream.last){
		// if(!axis_in.empty()) {
		tmp_stream = axis_in.read();
		stream_data_t tmp_data = tmp_stream.data;

		data_fifo_in.write(tmp_data);
		// }
	}
}

void stream_samples_out(hls::stream<stream_data_t, STREAM_BEATS_PER_TILE> &data_fifo,
						hls::stream<axis_t> &out_stream){
	
	axis_t tmp_stream;
	int out_ptr = 0;
	int fmap_ptr_abs = 0;
	for(out_ptr = 0; out_ptr < STREAM_BEATS_PER_TILE; out_ptr++){
		tmp_stream.last = (out_ptr == (STREAM_BEATS_PER_TILE - 1));

		// 128 bits = 16 bytes = 16 bits of keep
		// Max width is 128 bits
		// Fewer bytes will truncate
		tmp_stream.keep = 0xffff;
		tmp_stream.strb = 0xffff;
		tmp_stream.data = data_fifo.read();

		out_stream.write(tmp_stream);
	}
}

// Read from the input FIFO and convert the data to fixed point format
void fill_img_arr(hls::stream<axis_t> &input_fifo,
				  fixed_4_8_t img_in[INPUT_HEIGHT_PIX][INPUT_WIDTH_PIX][BYTES_PER_PIXEL]){

	int fmap_ptr_abs = 0;
	FILL_IMG_ARR: 
	for(int out_ptr = 0; out_ptr < STREAM_BEATS_PER_TILE; out_ptr++){
		axis_t tmp_strm = input_fifo.read();
		stream_data_t tmp_data = tmp_strm.data;

		for(int pixel_no = 0; pixel_no < 4; pixel_no++){
			for(int byte = 0; byte < 3; byte++){
				int row_idx = (fmap_ptr_abs / (INPUT_WIDTH_PIX * BYTES_PER_PIXEL));
				int col_idx = (fmap_ptr_abs / BYTES_PER_PIXEL) % INPUT_WIDTH_PIX;
				int chn_idx =  fmap_ptr_abs % BYTES_PER_PIXEL;
	
				int hi = (pixel_no * 4 * 8) + ((byte + 1) * 8) - 1;
				int low = hi - 7;
				fixed_9_8_t tmp_98 = tmp_data.range(hi, low);
				fixed_4_8_t tmp_val = tmp_98 >> 8;

				img_in[row_idx][col_idx][chn_idx] = tmp_val;
	
				fmap_ptr_abs++;
			}
		}
	}
}

void fill_output_fifo(fixed_4_8_t img_in[INPUT_HEIGHT_PIX][INPUT_WIDTH_PIX][BYTES_PER_PIXEL],
					  hls::stream<stream_data_t, STREAM_BEATS_PER_TILE> &output_fifo){

	int fmap_ptr_abs = 0;
	WRITE_OUTPUT:for(int out_ptr = 0; out_ptr < STREAM_BEATS_PER_TILE; out_ptr++){
#pragma HLS PIPELINE II=20
		stream_data_t tmp_data;
		for(int i = 0; i < BYTES_PER_TRANSFER; i++){
			// #pragma HLS UNROLL factor=BYTES_PER_TRANSFER

			int row_idx = (fmap_ptr_abs / (INPUT_WIDTH_PIX * BYTES_PER_PIXEL));
			int col_idx = (fmap_ptr_abs / BYTES_PER_PIXEL) % INPUT_WIDTH_PIX;
			int chn_idx =  fmap_ptr_abs % BYTES_PER_PIXEL;

			fixed_4_8_t tmp_val = img_in[row_idx][col_idx][chn_idx];
			tmp_data.range(8 * (i + 1) - 1, 8 * i) = tmp_val.range(7, 0);

			// Cast to fixed 9_8, multiply by 256, cast to uint8_t
			// fixed_9_8_t tmp_val = img_in[row_idx][col_idx][chn_idx];
			// tmp_val = tmp_val << 8;
			// uint8_t bits = (uint8_t)tmp_val.to_uint();

			fmap_ptr_abs++;
		}
		output_fifo.write(tmp_data);
	}
}

/**
 * Load the tile_in array with values, accounting for padding (2x2)
 */
void prep_tile(hls::stream<axis_t> &in_stream, hls::stream<fixed_4_8_t, IN_PADDED_SIZE*IN_PADDED_SIZE> tile_in[3]){
	axis_t tmp_stream;
	tmp_stream.last = 0;

	// Fill the first two rows with zeros
	PAD_TOP:
	for(int i = 0; i < IN_PADDED_SIZE * 2; i++){
		tile_in[0].write(0);
		tile_in[1].write(0);
		tile_in[2].write(0);
	}

	// Do the 28 rows of pixels
	READ_ROWS:
	for(int row = 0; row < INPUT_HEIGHT_PIX; row++){

		// Pad the first two pixels with zeros
		tile_in[0].write(0); tile_in[0].write(0);
		tile_in[1].write(0); tile_in[1].write(0);
		tile_in[2].write(0); tile_in[2].write(0);

		// Fill the meat and potatos 
		for(int beat = 0; beat < BEATS_PER_ROW; beat++){
		#pragma HLS PIPELINE II=1
			tmp_stream = in_stream.read();
			
			// 4 pixels per transfer from a 128-bit stream
			stream_data_t tmp_data = tmp_stream.data;

			// Pixel 0
			fixed_9_8_t r0 = tmp_data.range(7, 0);
			fixed_9_8_t g0 = tmp_data.range(15, 8);
			fixed_9_8_t b0 = tmp_data.range(23, 16);
			// Discard tmp_data.range(31, 24)
	
			// Pixel 1
			fixed_9_8_t r1 = tmp_data.range(39, 32);
			fixed_9_8_t g1 = tmp_data.range(47, 40);
			fixed_9_8_t b1 = tmp_data.range(55, 48);
			// Discard tmp_data.range(63, 56)
	
			// Pixel 2
			fixed_9_8_t r2 = tmp_data.range(71, 64);
			fixed_9_8_t g2 = tmp_data.range(79, 72);
			fixed_9_8_t b2 = tmp_data.range(87, 80);
			// Discard tmp_data.range(95, 88)
	
			// Pixel 3
			fixed_9_8_t r3 = tmp_data.range(103, 96);
			fixed_9_8_t g3 = tmp_data.range(111, 104);
			fixed_9_8_t b3 = tmp_data.range(119, 112);
			// Discard tmp_data.range(127, 120)

			// Divide by 256, cast to 12 bit fixed, write to FIFO
			tile_in[0].write((fixed_4_8_t)(r0 >> 8)); 
			tile_in[0].write((fixed_4_8_t)(r1 >> 8)); 
			tile_in[0].write((fixed_4_8_t)(r2 >> 8)); 
			tile_in[0].write((fixed_4_8_t)(r3 >> 8));

			tile_in[1].write((fixed_4_8_t)(g0 >> 8)); 
			tile_in[1].write((fixed_4_8_t)(g1 >> 8)); 
			tile_in[1].write((fixed_4_8_t)(g2 >> 8)); 
			tile_in[1].write((fixed_4_8_t)(g3 >> 8));

			tile_in[2].write((fixed_4_8_t)(b2 >> 8)); 
			tile_in[2].write((fixed_4_8_t)(b1 >> 8)); 
			tile_in[2].write((fixed_4_8_t)(b2 >> 8)); 
			tile_in[2].write((fixed_4_8_t)(b3 >> 8));
		}
	
		// Pad the last two pixels with zeros
		tile_in[0].write(0); tile_in[0].write(0);
		tile_in[1].write(0); tile_in[1].write(0);
		tile_in[2].write(0); tile_in[2].write(0);
	}
	
	// Fill the last two rows with zeros
	PAD_BOTTOM:
	for(int i = 0; i < IN_PADDED_SIZE * 2; i++){
		tile_in[0].write(0);
		tile_in[1].write(0);
		tile_in[2].write(0);
	}
}

fixed_4_8_t perform_mac5(const fixed_4_8_t weights[5], fixed_4_8_t slider[5]){
	#pragma HLS INLINE
	fixed_4_8_t sum = 0.0;
	DO_MAC5:
	for(int w = 0; w < 5; w++){
		#pragma HLS UNROLL
		sum += weights[w] * slider[w];
	}
	return sum;
}

fixed_4_8_t prelu(const fixed_4_8_t weight, fixed_4_8_t value){
	#pragma HLS INLINE
	if(value >= 0) return value;
	else return value * weight;
}


void print_slider5(fixed_4_8_t slider[IN_CHN_LAYER_1][5]){
	printf("[ ");
	for(int i = 0; i < 5; i++){
		printf("%8.6f ",slider[0][i].to_float());
	}
	printf("] ");
}

/**
 * Perform feature extraction convolutional layer. Assumes input feature map (tile_in) is appropriately padded
 */
void conv_extraction(hls::stream<fixed_4_8_t, IN_PADDED_SIZE*IN_PADDED_SIZE> tile_in[IN_CHN_LAYER_1], 
					 hls::stream<fixed_4_8_t, 28*28> map_out[OUT_CHN_LAYER_1]){


	// 3 input channels, 5 weights
	fixed_4_8_t slider[IN_CHN_LAYER_1][5];
	#pragma HLS array_partition variable=slider dim=0 type=complete
	#pragma array_partition variable=slider dim=1 type=complete

	hls::stream<fixed_4_8_t, 32> psum1[OUT_CHN_LAYER_1][IN_CHN_LAYER_1], psum2[OUT_CHN_LAYER_1][IN_CHN_LAYER_1], psum3[OUT_CHN_LAYER_1][IN_CHN_LAYER_1], psum4[OUT_CHN_LAYER_1][IN_CHN_LAYER_1];
//	#pragma HLS BIND_STORAGE variable=psum1 type=bram
	#pragma HLS array_partition variable=psum1 dim=0 type=complete

	for(int row = 0; row < IN_PADDED_SIZE; row++){

		// Prep the slider
		for(int ch = 0; ch < IN_CHN_LAYER_1; ch++){
			#pragma HLS UNROLL
			for(int idx = 0; idx < 4; idx++){
				#pragma HLS PIPELINE II=1
				slider[ch][idx] = tile_in[ch].read();
			}
		}

		// printf("Row %d\n", row);
		// Go across the column
		for(int col = 4; col < IN_PADDED_SIZE; col++){
			#pragma HLS PIPELINE II=1
			
			// Reset the final sum for each filter
			fixed_4_8_t final_sum[OUT_CHN_LAYER_1];
			#pragma HLS array_partition variable=final_sum dim=0 type=complete
			for(int filter = 0; filter < OUT_CHN_LAYER_1; filter++){
				#pragma HLS UNROLL
				final_sum[filter] = 0.0;
			}

			// Read the next slider value
			for(int ch = 0; ch < IN_CHN_LAYER_1; ch++){
				#pragma HLS UNROLL
				slider[ch][4] = tile_in[ch].read();
			}
			for(int filter = 0; filter < OUT_CHN_LAYER_1; filter++){
				// #pragma HLS UNROLL factor=2
				
				for(int ch = 0; ch < IN_CHN_LAYER_1; ch++){
					#pragma HLS UNROLL
	
					fixed_4_8_t mac0, mac1, mac2, mac3, mac4;
					fixed_4_8_t row1_psum, row2_psum, row3_psum, row4_psum;
	
					// print_slider5(slider); printf("\n");
	
					if(row < 28) {
						mac0 = perform_mac5(extraction_weights[filter][ch][0], slider[ch]);
						psum1[filter][ch].write(mac0);
					}
					if(row >= 1 && row < 29){
						row1_psum = psum1[filter][ch].read();
						mac1 = perform_mac5(extraction_weights[filter][ch][1], slider[ch]);
						psum2[filter][ch].write(row1_psum  + mac1);
					}
					if(row >= 2 && row < 30){
						row2_psum = psum2[filter][ch].read();
						mac2 = perform_mac5(extraction_weights[filter][ch][2], slider[ch]);
						psum3[filter][ch].write(row2_psum  + mac2);
					}
					if(row >= 3 && row < 31){
						row3_psum = psum3[filter][ch].read();
						mac3 = perform_mac5(extraction_weights[filter][ch][3], slider[ch]);
						psum4[filter][ch].write(row3_psum  + mac3);
					}
					if(row >= 4){
						row4_psum = psum4[filter][ch].read();
						mac4 = perform_mac5(extraction_weights[filter][ch][4], slider[ch]);
						fixed_4_8_t pre_activation = row4_psum + mac4;
						final_sum[filter] += pre_activation;
					}
				}
				if(row >= 4) map_out[filter].write(prelu(conv_extraction_prelu[filter], final_sum[filter] + conv_bias_extraction[filter]));
			} // For every filter in the layer

			// Shift the slider
			for(int ch = 0; ch < IN_CHN_LAYER_1; ch++){
				#pragma HLS UNROLL
				slider[ch][0] = slider[ch][1];
				slider[ch][1] = slider[ch][2];
				slider[ch][2] = slider[ch][3];
				slider[ch][3] = slider[ch][4];
			} 
		} // For every column in the row
	} // For every row in the tile
}

void conv2d_top(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream){
	#pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return
	
	// 1. Load the image into 3 separate streams (FIFOs), converting to fixed_4_8_t on the fly
	hls::stream<fixed_4_8_t, IN_PADDED_SIZE*IN_PADDED_SIZE> tile_in[3];
//	#pragma HLS BIND_STORAGE variable=tile_in type=bram
	#pragma HLS array_partition variable=tile_in dim=0 type=complete
	hls::stream<fixed_4_8_t, 28*28> map_extraction[OUT_CHN_LAYER_1];

//	#pragma HLS DATAFLOW

	prep_tile(in_stream, tile_in);

//	for(int i = 0; i < 5; i++){
//		for(int j = 0; j < 32; j++){
//			printf("%10.6f ", tile_in[0].read().to_float());
//		}
//		printf("\n");
//	}
//	return;

	conv_extraction(tile_in, map_extraction);

	// std::cout<<"INFO [conv2d] Stream size: "<< map_extraction[0].size()<<std::endl;
	// for(int i = 0; i < 28; i++){
	// 	std::cout<<"INFO [conv2d] Value from Conv: "<< map_extraction[0].read().to_float() << std::endl;
	// }
	// std::cout<<"INFO [conv2d] Stream size: "<< map_extraction[0].size()<<std::endl;
	for(int i = 0; i < 5; i++){
		printf("INFO [conv2d] First row of conv from feature map %d:\n", i);
		for (int col = 0; col < 28; col++){
			printf("%9.6f ", map_extraction[i].read().to_float());
		}
		printf("\n");
	}
}



/*

Pseudo code for convolution operation with FIFOs
 - Each input channel will be in its own FIFO
 - There will be (kernel_height - 1) FIFOs in each processing element
 - For a 28 x 28 input tile with a kernel of 5, there will be 2x2 padding to make a 32 x 32 tile
 - For design simplicity, even though zero padding is used, just pretend that it's a 32x32 tile and 
   all convolution happens inside. i.e. no edge or corner cases

   

 */
