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

//	stream_data_t mem[STREAM_BEATS_PER_TILE];
	// Rows x Cols x Channels
	fixed_4_8_t img_in[INPUT_HEIGHT_PIX][INPUT_WIDTH_PIX][BYTES_PER_PIXEL];

	// Will hang if the stream is empty and there's no tlast
	int channel = 0;
	int row = 0;
	int col = 0;
	int i;
	while(!tmp_stream.last){
		if(!in_stream.empty()) {
			tmp_stream = in_stream.read();
			stream_data_t tmp_data = tmp_stream.data;
			for(i = 0; i < 16; i++){
				// #pragma HLS UNROLL factor=16
				fixed_9_8_t tmp_val = tmp_data.range(8 * (i + 1) - 1, 8 * i);
				fixed_4_8_t tmp_val_fixed = tmp_val >> 8;
				img_in[row][col][channel] = tmp_val_fixed;
				channel = (channel + 1) % BYTES_PER_PIXEL;
				if(channel == 0){
					col++;
					if(col == INPUT_WIDTH_PIX){
						col = 0;
						row++;
						if(row == INPUT_HEIGHT_PIX){
							row = 0;
						}
					}
				}
			}
		}
	}

	// For debug, print the first 32 values in img_in. They should be float representations of the input image
	for(i = 0; i < 32; i++){
		std::cout << img_in[i / (INPUT_WIDTH_PIX * BYTES_PER_PIXEL)][(i / BYTES_PER_PIXEL) % INPUT_WIDTH_PIX][i % BYTES_PER_PIXEL] << std::endl;
	}

	// Turn each 8 bit sample to a fixed point representation

	int out_ptr = 0;
	for(out_ptr = 0; out_ptr < STREAM_BEATS_PER_TILE; out_ptr++){
		tmp_stream.last = (out_ptr == (STREAM_BEATS_PER_TILE - 1));

		// 128 bits = 16 bytes = 16 bits of keep
		// Max width is 128 bits
		tmp_stream.keep = 0xffff;
		tmp_stream.strb = 0xffff;
//		tmp_stream.data = mem[out_ptr];

		out_stream.write(tmp_stream);
	}
}
