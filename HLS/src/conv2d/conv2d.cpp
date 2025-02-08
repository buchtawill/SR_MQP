#include "conv2d.h"
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <iostream>

void conv2d_top(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream){
	#pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return

	axis_t tmp_stream;
	tmp_stream.last = 0;

	stream_data_t mem[STREAM_BEATS_PER_TILE];

	int in_ptr = 0;
	while(!tmp_stream.last){
		tmp_stream = in_stream.read();
		mem[in_ptr] = tmp_stream.data;
		in_ptr++;
	}

//	std::cout<<"We have read the whole input image"<<std::endl;

	int out_ptr = 0;
	for(out_ptr = 0; out_ptr < STREAM_BEATS_PER_TILE; out_ptr++){
		tmp_stream.last = (out_ptr == (STREAM_BEATS_PER_TILE - 1));

		// 128 bits = 16 bytes = 16 bits of keep
		tmp_stream.keep = 0xffff;
		tmp_stream.strb = 0xffff;
		tmp_stream.data = mem[out_ptr];

		out_stream.write(tmp_stream);
	}
}
