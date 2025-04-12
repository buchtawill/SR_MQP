#ifndef CONV2D_H
#define CONV2D_H

#include <ap_axi_sdata.h>
#include "hls_stream.h"
#include "ap_int.h"
#include "ap_fixed.h"

#define STREAM_WIDTH            128 // Must be a power of 2 greater than 8
#define INPUT_WIDTH_PIX         32
#define INPUT_HEIGHT_PIX        32
#define BYTES_PER_PIXEL         4
#define BYTES_PER_TRANSFER      (STREAM_WIDTH / 8)
#define BEATS_PER_ROW           (INPUT_WIDTH_PIX * BYTES_PER_PIXEL) / BYTES_PER_TRANSFER
#define NUM_TRANSFER_BYTES      (INPUT_WIDTH_PIX * INPUT_HEIGHT_PIX * BYTES_PER_PIXEL)
#define STREAM_BEATS_PER_TILE   ((NUM_TRANSFER_BYTES * 8) / STREAM_WIDTH)

#define IN_CHN_LAYER_FEATURE_EXTRACTION0    3
#define OUT_CHN_LAYER_FEATURE_EXTRACTION0   16
#define NUM_PE_LAYER_FEATURE_EXTRACTION0    2

#define IN_CHN_LAYER_SHRINK0    16
#define OUT_CHN_LAYER_SHRINK0   12
#define NUM_PE_LAYER_SHRINK0    6

#define IN_CHN_LAYER_MAP0    12
#define OUT_CHN_LAYER_MAP0   12
#define NUM_PE_LAYER_MAP0    2

#define IN_CHN_LAYER_MAP2    12
#define OUT_CHN_LAYER_MAP2   12
#define NUM_PE_LAYER_MAP2    2

#define IN_CHN_LAYER_MAP4    12
#define OUT_CHN_LAYER_MAP4   8
#define NUM_PE_LAYER_MAP4    2

#define IN_CHN_LAYER_EXPAND0    8
#define OUT_CHN_LAYER_EXPAND0   8
#define NUM_PE_LAYER_EXPAND0    4

#define IN_CHN_LAYER_DECONV0    8
#define OUT_CHN_LAYER_DECONV0   3
#define NUM_PE_LAYER_DECONV0    1

typedef ap_uint<STREAM_WIDTH> stream_data_t;

// TODO WARNING NOTE: Using AP_SAT can cost up to a 20% increase in LUT usage!!!
// After development and debug, change to AP_WRAP
// Total bit width, integer bits, Quant mode, Overflow mode
typedef ap_fixed<24, 9, AP_RND_ZERO, AP_WRAP> fixed_9_8_t;

// 4 bits int (including sign), 8 bits fractional
// total bits, int bits
typedef ap_fixed<18, 6, AP_RND_ZERO, AP_WRAP> fixed_4_8_t;
// typedef ap_fixed<36, 10, AP_RND_ZERO, AP_WRAP> fixed_4_8_t;
// typedef float fixed_4_8_t;

// Define axis_t with data width of 8 bits and no additional signals
typedef hls::axis<stream_data_t, 0, 0, 0> axis_t;

typedef hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*INPUT_HEIGHT_PIX> ch_stream_t;
typedef hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*INPUT_HEIGHT_PIX * 2 * 2> upscaled_stream_t;

void conv2d_top(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream);


#endif // CONV2D_H
