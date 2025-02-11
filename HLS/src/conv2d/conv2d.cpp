#include "conv2d.h"
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <iostream>
#include <stdio.h>

void conv2d_top(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream){
	#pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return

	axis_t tmp_stream;
	tmp_stream.last = 0;

	// Rows x Cols x Channels
	fixed_4_8_t img_in[INPUT_HEIGHT_PIX][INPUT_WIDTH_PIX][BYTES_PER_PIXEL];
	#pragma HLS BIND_STORAGE variable=img_in type=ram_2p impl=bram
	
#pragma HLS ARRAY_PARTITION variable=img_in complete dim=3
// Optionally, you could also consider reshaping another dimension if needed:
// #pragma HLS ARRAY_RESHAPE variable=img_in cyclic factor=2 dim=2


	int in_ptr_abs = 0;
	while(!tmp_stream.last){
// #pragma HLS PIPELINE II=1
		if(!in_stream.empty()) {
			tmp_stream = in_stream.read();
			stream_data_t tmp_data = tmp_stream.data;

			// For each byte in the input stream data
			for(int i = 0; i < BYTES_PER_TRANSFER; i++){
				 #pragma HLS UNROLL

				fixed_9_8_t tmp_val = tmp_data.range(8 * (i + 1) - 1, 8 * i);
				fixed_4_8_t tmp_val_fixed = tmp_val >> 8;

				int row_idx = (in_ptr_abs / (INPUT_WIDTH_PIX * BYTES_PER_PIXEL));
				int col_idx = (in_ptr_abs / BYTES_PER_PIXEL) % INPUT_WIDTH_PIX;
				int chn_idx =  in_ptr_abs % BYTES_PER_PIXEL;
				
				img_in[row_idx][col_idx][chn_idx] = tmp_val_fixed;
				in_ptr_abs++;
			}
		}
	}

	// For debug, print the first 32 values in img_in. They should be float representations of the input image
	// for(i = 0; i < 32; i++){
	// 	std::cout << img_in[i / (INPUT_WIDTH_PIX * BYTES_PER_PIXEL)][(i / BYTES_PER_PIXEL) % INPUT_WIDTH_PIX][i % BYTES_PER_PIXEL] << std::endl;
	// }

	// Convert them all back to uint8 represenations (cast to 9_8, multiply by 256, cast to uint8_t
	int out_ptr = 0;
	int fmap_ptr_abs = 0;
	for(out_ptr = 0; out_ptr < STREAM_BEATS_PER_TILE; out_ptr++){
// #pragma HLS PIPELINE II=1
		tmp_stream.last = (out_ptr == (STREAM_BEATS_PER_TILE - 1));

		// 128 bits = 16 bytes = 16 bits of keep
		// Max width is 128 bits
		// Fewer bytes will truncate
		tmp_stream.keep = 0xffff;
		tmp_stream.strb = 0xffff;
		stream_data_t tmp_data;

		for(int i = 0; i < BYTES_PER_TRANSFER; i++){
			 #pragma HLS UNROLL

			int row_idx = (fmap_ptr_abs / (INPUT_WIDTH_PIX * BYTES_PER_PIXEL));
			int col_idx = (fmap_ptr_abs / BYTES_PER_PIXEL) % INPUT_WIDTH_PIX;
			int chn_idx =  fmap_ptr_abs % BYTES_PER_PIXEL;

			// Cast to fixed 9_8, multiply by 256, cast to uint8_t
			fixed_9_8_t tmp_val = img_in[row_idx][col_idx][chn_idx];
			tmp_val = tmp_val << 8;
			uint8_t bits = (uint8_t)tmp_val.to_uint();

			tmp_data.range(8 * (i + 1) - 1, 8 * i) = bits;
			fmap_ptr_abs++;
		}
		tmp_stream.data = tmp_data;

		out_stream.write(tmp_stream);
	}
}



