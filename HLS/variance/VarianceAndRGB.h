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

// Override mode definitions
#define OVERRIDE_MODE_DEFAULLT 0
#define OVERRIDE_MODE_CONV 1
#define OVERRIDE_MODE_INTERP 2

typedef ap_uint<8> pixel_component; // 8-bit per channel for each pixel
typedef ap_uint<32> data_stream;

// Function prototype
void process_tile(hls::stream<data_stream> &pixel_stream_in,
                  hls::stream<data_stream> &conv_out,
                  hls::stream<data_stream> &interp_out,
                  float threshold,
                  ap_uint<2> override_mode);

#endif // VARIANCE_AND_RGB_H
