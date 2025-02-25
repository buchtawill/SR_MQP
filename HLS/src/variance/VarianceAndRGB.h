#ifndef VARIANCE_AND_RGB_H
#define VARIANCE_AND_RGB_H

#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <iostream>
#include <ap_axi_sdata.h>

// tile dimensions
#define TRANSFER_WIDTH 128
#define YUYV_BITS_PER_PIXEL 16
#define RGB_BITS_PER_PIXEL 32
#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define PIXEL_COUNT (IMAGE_WIDTH * IMAGE_HEIGHT)
#define YUYV_NUM_TRANSFERS PIXEL_COUNT / (TRANSFER_WIDTH / YUYV_BITS_PER_PIXEL)
#define RGB_NUM_TRANSFERS PIXEL_COUNT / (TRANSFER_WIDTH / RGB_BITS_PER_PIXEL)

#define RGB_PAD 0b00000000
#define RGB_MIN ((fixed_pixel)0)
#define RGB_MAX ((fixed_pixel)255)

// Override mode definitions
#define OVERRIDE_MODE_DEFAULT 0
#define OVERRIDE_MODE_CONV 1
#define OVERRIDE_MODE_INTERP 2

typedef ap_uint<8> ap_8; // 8-bit per channel for each pixel
typedef ap_uint<32> ap_32;
typedef ap_uint<128> ap_uint_128; // 128-bit YUYV pixel stream (8 pixels)
typedef ap_fixed<32, 9> fixed_pixel; // for channel conversion math
typedef ap_fixed<32, 24> fixed_32; // fixed point instead of float

// Define axis_t with data width of 128 bits and no additional signals
typedef hls::axis<ap_uint_128, 0, 0, 0> axis_t;

// Function prototype
void process_tile(hls::stream<axis_t> &pixel_stream_in,
                  hls::stream<axis_t> &conv_out,
                  hls::stream<axis_t> &interp_out,
                  unsigned int threshold,
                  ap_uint<2> override_mode);

void rgb_convert(ap_uint_128 *RGB_pixel_data, ap_uint_128 *pixel_data);

#endif // VARIANCE_AND_RGB_H
