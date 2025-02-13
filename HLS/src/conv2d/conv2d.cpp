#include "conv2d.h"
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <iostream>
#include <stdio.h>

void fill_input_fifo(hls::stream<axis_t> &axis_in,
					 hls::stream<stream_data_t, STREAM_BEATS_PER_TILE> &data_fifo_in){

	axis_t tmp_stream;
	tmp_stream.last = 0;

	int in_ptr_abs = 0;
	while(!tmp_stream.last){
		if(!axis_in.empty()) {
			tmp_stream = axis_in.read();
			stream_data_t tmp_data = tmp_stream.data;

			data_fifo_in.write(tmp_data);
		}
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
void fill_img_arr(hls::stream<stream_data_t, STREAM_BEATS_PER_TILE> &input_fifo,
				  fixed_4_8_t img_in[INPUT_HEIGHT_PIX][INPUT_WIDTH_PIX][BYTES_PER_PIXEL]){

	int fmap_ptr_abs = 0;
//	stream_data_t tmp_1 = input_fifo.read();
	FILL_IMG_ARR: for(int out_ptr = 0; out_ptr < STREAM_BEATS_PER_TILE; out_ptr++){
#pragma HLS PIPELINE II=20
		stream_data_t tmp_2 = input_fifo.read();
		for(int i = 0; i < BYTES_PER_TRANSFER; i++){
			// #pragma HLS UNROLL factor=BYTES_PER_TRANSFER
	
			int row_idx = (fmap_ptr_abs / (INPUT_WIDTH_PIX * BYTES_PER_PIXEL));
			int col_idx = (fmap_ptr_abs / BYTES_PER_PIXEL) % INPUT_WIDTH_PIX;
			int chn_idx =  fmap_ptr_abs % BYTES_PER_PIXEL;

			fixed_9_8_t tmp_98 = tmp_2.range(8 * (i + 1) - 1, 8 * i);
			fixed_4_8_t tmp_val = tmp_98 >> 8;
			img_in[row_idx][col_idx][chn_idx] = tmp_val;
	
			fmap_ptr_abs++;
		}
//		tmp_1 = tmp_2;
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

void conv2d_top(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream){
	#pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return
	
	// Rows x Cols x Channels
	fixed_4_8_t img_in[INPUT_HEIGHT_PIX][INPUT_WIDTH_PIX][BYTES_PER_PIXEL];
	#pragma HLS BIND_STORAGE variable=img_in type=ram_2p impl=bram

	// Create a FIFO to store tile data
	hls::stream<stream_data_t, STREAM_BEATS_PER_TILE> input_fifo;
	hls::stream<stream_data_t, STREAM_BEATS_PER_TILE> output_fifo;
#pragma HLS BIND_STORAGE variable=input_fifo  type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=output_fifo type=ram_2p impl=bram

// #pragma HLS DATAFLOW

	// These two will happen concurrently
	fill_input_fifo(in_stream, input_fifo);
	fill_img_arr(input_fifo, img_in);

	// Do some processing here

	// Write the output to the output FIFO
	fill_output_fifo(img_in, output_fifo);
	stream_samples_out(output_fifo, out_stream);
}



