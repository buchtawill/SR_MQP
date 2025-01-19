#ifndef BILINEAR_INTERPOLATION_H
#define BILINEAR_INTERPOLATION_H

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
#define BITS_PER_TRANSFER 32
#define BITS_PER_PIXEL 8
#define NUM_TRANSFERS (WIDTH_IN*HEIGHT_IN*CHANNELS*BITS_PER_PIXEL/BITS_PER_TRANSFER) //for 28x28x3 this is 588

// Define pixel_t as an 8-bit unsigned integer
typedef ap_uint<BITS_PER_PIXEL> pixel_t; // 8-bit per channel for each pixel
typedef ap_uint<BITS_PER_TRANSFER> data_streamed;

// Define axis_t with data width of 32 bits and no additional signals
typedef hls::axis<data_streamed, 0, 0, 0> axis_t;


// Function declaration for Bilinear Interpolation calculations
void bilinear_interpolation_calculations(pixel_t image_in[HEIGHT_IN][WIDTH_IN][CHANNELS],
                                         pixel_t image_out[HEIGHT_OUT][WIDTH_OUT][CHANNELS]);

// Function declaration for the main bilinear interpolation function
void bilinear_interpolation_v2(hls::stream<axis_t> &in_stream,
                             hls::stream<axis_t> &out_stream);

// Testbench function declaration
void testbench();

#endif // BILINEAR_INTERPOLATION_H
