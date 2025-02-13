#ifndef CONV2D_H
#define CONV2D_H

#include <ap_axi_sdata.h>
#include "hls_stream.h"
#include "ap_int.h"
#include "ap_fixed.h"

#define STREAM_WIDTH            128 // Must be a power of 2 greater than 8
#define INPUT_WIDTH_PIX         28
#define INPUT_HEIGHT_PIX        28
#define BYTES_PER_PIXEL         3
#define NUM_TRANSFER_BYTES      (INPUT_WIDTH_PIX * INPUT_HEIGHT_PIX * BYTES_PER_PIXEL)
#define BYTES_PER_TRANSFER      (STREAM_WIDTH / 8)
#define STREAM_BEATS_PER_TILE   ((NUM_TRANSFER_BYTES * 8) / STREAM_WIDTH)

typedef ap_uint<STREAM_WIDTH> stream_data_t;


// TODO WARNING NOTE: Using AP_SAT can cost up to a 20% increase in LUT usage!!!
// After development and debug, change to AP_WRAP
// Total bit width, integer bits, Quant mode, Overflow mode
typedef ap_fixed<17, 9, AP_RND_ZERO, AP_WRAP> fixed_9_8_t;

// 4 bits int (including sign), 8 bits fractional
typedef ap_fixed<12, 4, AP_RND_ZERO, AP_WRAP> fixed_4_8_t;

// Define axis_t with data width of 8 bits and no additional signals
typedef hls::axis<stream_data_t, 0, 0, 0> axis_t;

void conv2d_top(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream);

#endif // CONV2D_H
