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

// Define pixel_t as an 8-bit unsigned integer (ap_uint<8> from HLS)
typedef ap_uint<32> pixel_t; // 8-bit per channel for each pixel

typedef hls::axis<pixel_t, 0, 0, 0> axis_t;

/*
// Define the AXI-stream interface for input and output
struct axi_stream {
    pixel_t data;  // Data representing a pixel value
    bool last;     // To indicate the last element in the stream
}; */

// Function declaration for Bilinear Interpolation calculations
void bilinear_interpolation_calculations(pixel_t image_in[HEIGHT_IN][WIDTH_IN][CHANNELS],
                                         pixel_t image_out[HEIGHT_OUT][WIDTH_OUT][CHANNELS]);

// Function declaration for the main bilinear interpolation function
void bilinear_interpolation(hls::stream<axis_t> &in_stream,
                             hls::stream<axis_t> &out_stream);

// Testbench function declaration
void testbench();

#endif // BILINEAR_INTERPOLATION_H
