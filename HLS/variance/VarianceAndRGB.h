#ifndef VARIANCE_AND_RGB_H
#define VARIANCE_AND_RGB_H

#include <hls_stream.h>
#include <ap_int.h>

// tile dimensions
#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define PIXEL_COUNT (IMAGE_WIDTH * IMAGE_HEIGHT)

// Override mode definitions
#define OVERRIDE_MODE_DEFAULLT 0
#define OVERRIDE_MODE_CONV 1
#define OVERRIDE_MODE_INTERP 2

// Function prototype
void process_tile(hls::stream<ap_uint<32>> &pixel_stream,
                  hls::stream<ap_uint<32>> &output_stream_high,
                  hls::stream<ap_uint<32>> &output_stream_low,
                  float threshold,
                  ap_uint<2> override_mode);

#endif // VARIANCE_AND_RGB_H
