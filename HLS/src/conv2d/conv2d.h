#ifndef CONV2D_H
#define CONV2D_H

#include <ap_axi_sdata.h>
#include "hls_stream.h"
#include "ap_int.h"
#include "ap_fixed.h"

#define STREAM_WIDTH            128 // Must be a power of 2 greater than 8
#define INPUT_WIDTH_PIX         28
#define INPUT_HEIGHT_PIX        28
#define BYTES_PER_PIXEL         4
#define BYTES_PER_TRANSFER      (STREAM_WIDTH / 8)
#define BEATS_PER_ROW           (INPUT_WIDTH_PIX * BYTES_PER_PIXEL) / BYTES_PER_TRANSFER
#define NUM_TRANSFER_BYTES      (INPUT_WIDTH_PIX * INPUT_HEIGHT_PIX * BYTES_PER_PIXEL)
#define STREAM_BEATS_PER_TILE   ((NUM_TRANSFER_BYTES * 8) / STREAM_WIDTH)

#define IN_CHN_LAYER_1          3
#define OUT_CHN_LAYER_1         44
#define NUM_PE_LAYER_1          44
#define IN_PADDED_SIZE          32

#define FEAT_EXT_PADDING 2
#define FEAT_EXT_PADDED_SIZE    (INPUT_WIDTH_PIX + 2*FEAT_EXT_PADDING)

typedef ap_uint<STREAM_WIDTH> stream_data_t;

// TODO WARNING NOTE: Using AP_SAT can cost up to a 20% increase in LUT usage!!!
// After development and debug, change to AP_WRAP
// Total bit width, integer bits, Quant mode, Overflow mode
typedef ap_fixed<17, 9, AP_RND_ZERO, AP_WRAP> fixed_9_8_t;

// 4 bits int (including sign), 8 bits fractional
// total bits, int bits
typedef ap_fixed<18, 5, AP_RND_ZERO, AP_WRAP> fixed_4_8_t;

// Define axis_t with data width of 8 bits and no additional signals
typedef hls::axis<stream_data_t, 0, 0, 0> axis_t;

void conv2d_top(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream);

const fixed_4_8_t conv_bias_extraction[44] = {
    0.0146,  0.0190,  0.0002,  0.0117, -0.0083,  0.0153, -0.0042, -0.0126,
     0.0144, -0.0129, -0.0040, -0.0017, -0.0195,  0.0217, -0.0023,  0.0185,
    -0.0176,  0.0032, -0.0055,  0.0332,  0.0142,  0.0031, -0.0357,  0.0090,
    -0.0125,  0.0146, -0.0095, -0.0201,  0.0114, -0.0168, -0.0142,  0.0051,
     0.0060,  0.0157, -0.0120,  0.0184,  0.0179, -0.0061,  0.0101,  0.0128,
     0.0062,  0.0106, -0.0149,  0.0020
};

const fixed_4_8_t conv_extraction_prelu[44] = {
    0.3086,  0.7313,  0.5082, -0.0311,  0.2554,  0.6756,  0.2276,  0.2919,
    0.5258,  0.2568,  1.4363,  0.4791,  0.3077,  1.0420,  1.0725,  1.0955,
    0.2495,  0.9166,  0.2436,  0.6043,  0.7211,  1.2348,  0.2671,  1.0052,
    0.2553,  0.3675,  0.2656,  0.3888,  0.2617,  0.2484,  0.2567,  0.2940,
    1.5282,  1.5432,  0.2662,  0.2995,  0.8394,  0.3161,  0.2736, -0.5784,
    0.2787,  0.3268,  0.2723,  0.2538
};

#endif // CONV2D_H
