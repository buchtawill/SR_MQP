#ifndef MY_HLS_FUNCTION_H
#define MY_HLS_FUNCTION_H
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>


#define MATRIX_WIDTH 28
#define MATRIX_HEIGHT 28
#define CHANNELS 3
#define BITS_PER_TRANSFER 32
#define BITS_PER_PIXEL 8
#define NUM_TRANSFERS (MATRIX_WIDTH*MATRIX_HEIGHT*CHANNELS*BITS_PER_PIXEL/BITS_PER_TRANSFER)


// Define pixel_t as an 8-bit unsigned integer
typedef ap_uint<8> pixel_t; // 8-bit per channel for each pixel
typedef ap_uint<32> data_streamed;

// Define axis_t with data width of 8 bits and no additional signals
typedef hls::axis<data_streamed, 0, 0, 0> axis_t;

void my_hls_function(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream);

#endif
