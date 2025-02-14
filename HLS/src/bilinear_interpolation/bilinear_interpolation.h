#ifndef BILINEAR_INTERPOLATION_H
#define BILINEAR_INTERPOLATION_H
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <iostream>
#include <cmath>


#define WIDTH_IN 28
#define HEIGHT_IN 28
#define CHANNELS 3
#define SCALE_FACTOR 2
#define WIDTH_OUT (WIDTH_IN * SCALE_FACTOR)
#define HEIGHT_OUT (HEIGHT_IN * SCALE_FACTOR)
#define PIXELS_IN (WIDTH_IN * HEIGHT_IN * CHANNELS)
#define PIXELS_OUT (WIDTH_OUT * HEIGHT_OUT * CHANNELS)
#define BITS_PER_TRANSFER 128
#define BITS_PER_PIXEL 8
#define NUM_TRANSFERS (WIDTH_IN*HEIGHT_IN*CHANNELS*BITS_PER_PIXEL/BITS_PER_TRANSFER)
#define NUM_TRANSFERS_OUT (NUM_TRANSFERS*SCALE_FACTOR*SCALE_FACTOR)
#define PIXELS_PER_TRANSFER (BITS_PER_TRANSFER / BITS_PER_PIXEL)


// Define pixel_t as an 8-bit unsigned integer
typedef ap_uint<8> pixel_t; // 8-bit per channel for each pixel
typedef ap_uint<BITS_PER_TRANSFER> data_streamed;
typedef ap_fixed<32, 20> fixed;

// Define axis_t with data width of 8 bits and no additional signals
typedef hls::axis<data_streamed, 0, 0, 0> axis_t;

// Function declaration for Bilinear Interpolation calculations
int bilinear_interpolation_calculations(pixel_t image_in[NUM_TRANSFERS],
                           	   	   	   	pixel_t image_out[NUM_TRANSFERS_OUT]);

void bilinear_interpolation(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream);

#endif
