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
#define IN_PADDED_SIZE          32

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

// Example weights - taken from Models/FSRCNN/extract_weights.py
const fixed_4_8_t conv_weights[3][5][5] = {
    {
        {-0.11433014, -0.06428213,  0.04952151, -0.02263108,  0.02250793 },
        {-0.03798941, -0.00633621, -0.07827316,  0.03095409,  0.01443817 },
        { 0.03689848, -0.09081642,  0.05860689,  0.06232064,  0.00874583 },
        { 0.00830404, -0.09028250,  0.04836075,  0.01574024, -0.05474688 },
        { 0.03897108, -0.06225232,  0.01753724,  0.02418811, -0.03822224 }
    },
    {
        { 0.00477085, -0.05933009, -0.04227761,  0.01911220, -0.06199792 },
        {-0.04209856, -0.06560600, -0.05108242,  0.08851463, -0.02541809 },
        { 0.03671201, -0.03489959, -0.01189760, -0.06215083,  0.01050324 },
        {-0.00799513,  0.00076000,  0.01511470, -0.10405967,  0.01873976 },
        {-0.04879151,  0.01140088,  0.04946414,  0.01681182, -0.00011618 }
    },
    {
        {-0.03623180,  0.02557710, -0.01484760, -0.01260940,  0.00872839 },
        {-0.00589560,  0.02435622, -0.06638044,  0.00577510,  0.00808473 },
        { 0.01797765, -0.00885874,  0.16383554,  0.02077497,  0.00002132 },
        {-0.00601372, -0.02807183,  0.08771470, -0.03528251,  0.05220633 },
        { 0.01051262,  0.00916464, -0.05010103,  0.00487509,  0.01671523 }
    }
};

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
