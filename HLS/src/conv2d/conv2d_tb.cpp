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

		// 4 pixels per transfer from a 128-bit stream
	
		// Pixel 0
		tmp_data.range(7, 0)   = coin_tile_low_res_rgb[coin_idx + 0];
		tmp_data.range(15, 8)  = coin_tile_low_res_rgb[coin_idx + 1];
		tmp_data.range(23, 16) = coin_tile_low_res_rgb[coin_idx + 2];
		// Discard tmp_data.range(31, 24)
		
		// Pixel 1
		tmp_data.range(39, 32) = coin_tile_low_res_rgb[coin_idx + 3];
		tmp_data.range(47, 40) = coin_tile_low_res_rgb[coin_idx + 4];
		tmp_data.range(55, 48) = coin_tile_low_res_rgb[coin_idx + 5];
		// Discard tmp_data.range(63, 56)
		
		// Pixel 2
		tmp_data.range(71, 64) = coin_tile_low_res_rgb[coin_idx + 6];
		tmp_data.range(79, 72) = coin_tile_low_res_rgb[coin_idx + 7];
		tmp_data.range(87, 80) = coin_tile_low_res_rgb[coin_idx + 8];
		// Discard tmp_data.range(95, 88)
		
		// Pixel 3
		tmp_data.range(103, 96)  = coin_tile_low_res_rgb[coin_idx + 9];
		tmp_data.range(111, 104) = coin_tile_low_res_rgb[coin_idx + 10];
		tmp_data.range(119, 112) = coin_tile_low_res_rgb[coin_idx + 11];
		coin_idx += 12;
		// Discard tmp_data.range(127, 120)

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
	// i = 0;
	// bool tlast = false;
	// bool failed = false;
	// do{
	// 	axis_t tmp_stream;

	// 	tmp_stream = out_stream.read();
	// 	tlast = tmp_stream.last;

	// 	// Check the data
	// 	for(j = 0; j < BYTES_PER_TRANSFER; j++){
	// 		coin_idx = (i * BYTES_PER_TRANSFER) + j;
	// 		uint8_t tmp_stream_val = tmp_stream.data.range((j+1)*8-1, j*8);
	// 		if(tmp_stream_val != coin_tile_low_res_rgb[coin_idx]){
	// 			printf("ERROR [conv2d_tb] Expected %3u, got %3u\n", coin_tile_low_res_rgb[coin_idx], tmp_stream_val);
	// 			failed = true;
	// 		}
	// 		else{
	// 			printf("GOOD [conv2d_tb] Expected %3u, got %3u\n", coin_tile_low_res_rgb[coin_idx], tmp_stream_val);
	// 		}
	// 	}
	// 	i++;
	// }while(!tlast);

	// if(failed) return -1;
	// else return 0;
}
