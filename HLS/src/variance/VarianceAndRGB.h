#ifndef VARIANCE_AND_RGB_H
#define VARIANCE_AND_RGB_H

#include <hls_stream.h>
#include <ap_int.h>
#include <iostream>
#include <ap_axi_sdata.h>

// tile dimensions
#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define PIXEL_COUNT (IMAGE_WIDTH * IMAGE_HEIGHT)
#define NUM_TRANSFERS PIXEL_COUNT / 8

// Override mode definitions
#define OVERRIDE_MODE_DEFAULT 0
#define OVERRIDE_MODE_CONV 1
#define OVERRIDE_MODE_INTERP 2

typedef ap_uint<8> pixel_component; // 8-bit per channel for each pixel
typedef ap_uint<32> YUYV32;
typedef ap_uint<128> data_stream; // 128-bit YUYV pixel stream (8 pixels)

// Define axis_t with data width of 32 bits and no additional signals
typedef hls::axis<data_stream, 0, 0, 0> axis_t;

// Function prototype
void process_tile(hls::stream<axis_t> &pixel_stream_in,
                  hls::stream<axis_t> &conv_out,
                  hls::stream<axis_t> &interp_out,
                  unsigned int threshold,
                  ap_uint<2> override_mode);

#endif // VARIANCE_AND_RGB_H
