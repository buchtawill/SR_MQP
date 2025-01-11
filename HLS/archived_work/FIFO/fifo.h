#ifndef FIFO_H
#define FIFO_H

#include <hls_stream.h>
#include <ap_int.h>
#include <iostream>
#include <ap_axi_sdata.h>

// Define the image dimensions and the scaling factor
#define WIDTH_IN 28
#define HEIGHT_IN 28
#define CHANNELS 3
#define SCALE_FACTOR 2
#define WIDTH_OUT (WIDTH_IN * SCALE_FACTOR)
#define HEIGHT_OUT (HEIGHT_IN * SCALE_FACTOR)

// Define pixel_t as an 8-bit unsigned integer (ap_uint<8> from HLS)
typedef ap_uint<8> pixel_t; // 8-bit per channel for each pixel

typedef hls::axis<pixel_t, 0, 0, 0> axis_t;


// Function declaration for the main bilinear interpolation function
void fifo(hls::stream<axis_t> &in_stream,
                             hls::stream<axis_t> &out_stream);

// Testbench function declaration
void testbench();

#endif
