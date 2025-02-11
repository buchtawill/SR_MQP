#include "conv2d.h"
#include "hls_stream.h"
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include "../image_coin_tile.h"
#include <iostream>
#include <stdio.h>
#include <stdint.h>

int main(){

	hls::stream<axis_t> in_stream("tb_input_stream");
	hls::stream<axis_t> out_stream("tb_output_stream");

	// Set up the input stream
	stream_data_t tmp_data;
	int i, j;
	int coin_idx = 0;
	for(i = 0; i < STREAM_BEATS_PER_TILE; i++){

		// Fill the tdata
		for(j = 0; j < BYTES_PER_TRANSFER; j++){

			uint16_t high_bit = (BYTES_PER_TRANSFER * 8) - 1;
			uint16_t low_bit  = high_bit - 7;
			coin_idx = (i * BYTES_PER_TRANSFER) + j;
			tmp_data >>= 8;
			tmp_data.range(high_bit,low_bit) = coin_tile_low_res_rgb[coin_idx];
		}

		// Write it to the stream
		axis_t tmp_stream;
		tmp_stream.last = (i == (STREAM_BEATS_PER_TILE - 1));
		tmp_stream.keep = 0xffff;
		tmp_stream.strb = 0xffff;
		tmp_stream.data = tmp_data;

		in_stream.write(tmp_stream);
	}

	// Run the conv2d
	conv2d_top(in_stream, out_stream);

	// Check the results
	i = 0;
	bool tlast = false;
	bool failed = false;
	do{
		axis_t tmp_stream;

		tmp_stream = out_stream.read();
		tlast = tmp_stream.last;

		// Check the data
		for(j = 0; j < BYTES_PER_TRANSFER; j++){
			coin_idx = (i * BYTES_PER_TRANSFER) + j;
			uint8_t tmp_stream_val = tmp_stream.data.range((j+1)*8-1, j*8);
			if(tmp_stream_val != coin_tile_low_res_rgb[coin_idx]){
				printf("ERROR [conv2d_tb] Expected %3u, got %3u\n", coin_tile_low_res_rgb[coin_idx], tmp_stream_val);
				failed = true;
			}
			else{
				printf("GOOD [conv2d_tb] Expected %3u, got %3u\n", coin_tile_low_res_rgb[coin_idx], tmp_stream_val);
			}
		}
		i++;
	}while(!tlast);

	if(failed) return -1;
	else return 0;
}
