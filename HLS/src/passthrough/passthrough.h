#ifndef PASSTHROUGH_H
#define PASSTHROUGH_H

#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>

// Define pixel_t as an 8-bit unsigned integer
typedef ap_uint<8> pixel_t; // 8-bit per channel for each pixel
typedef ap_uint<32> bytes_streamed;

// Define axis_t with data width of 8 bits and no additional signals
typedef hls::axis<bytes_streamed, 0, 0, 0> axis_t;

void passthrough(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream);

#endif
