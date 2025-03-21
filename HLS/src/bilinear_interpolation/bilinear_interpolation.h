#ifndef BILINEAR_INTERPOLATION_H
#define BILINEAR_INTERPOLATION_H
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <iostream>
#include <cmath>

#define WIDTH_IN 28
#define HEIGHT_IN 28
#define CHANNELS 3

#define SCALE_FACTOR 2
#define WIDTH_OUT (WIDTH_IN * SCALE_FACTOR)
#define HEIGHT_OUT (HEIGHT_IN * SCALE_FACTOR)
#define PIXELS_IN (WIDTH_IN * HEIGHT_IN)
#define PIXELS_OUT (WIDTH_OUT * HEIGHT_OUT)
#define BITS_PER_TRANSFER 128
#define BITS_PER_PIXEL 32
#define PIXELS_PER_TRANSFER (BITS_PER_TRANSFER/BITS_PER_PIXEL)
#define CHANNELS_PER_TRANSFER (PIXELS_PER_TRANSFER * CHANNELS)
#define BITS_PER_CHANNEL 8
#define NUM_TRANSFERS_IN (WIDTH_IN*HEIGHT_IN/PIXELS_PER_TRANSFER)
#define NUM_TRANSFERS_OUT (NUM_TRANSFERS_IN*SCALE_FACTOR*SCALE_FACTOR)

// Define pixel_t as an 8-bit unsigned integer
typedef ap_uint<24> pixel_t; // 8-bit per channel for each pixel
typedef ap_uint<8> channel_t;
typedef ap_uint<BITS_PER_TRANSFER> data_streamed;
typedef ap_fixed<32, 20> fixed;
typedef ap_uint<32> full_pixel;

// Define axis_t with data width of 8 bits and no additional signals
typedef hls::axis<data_streamed, 0, 0, 0> axis_t;

void bilinear_interpolation(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream);

std::vector<uint8_t> bilinearInterpolation(
    const std::vector<uint8_t>& image,
    int width,
    int height,
    int channels,
    float scale);

#define SLIDER_WIDTH_IN 7
#define SLIDER_HEIGHT_IN 7
#define SLIDER_WIDTH_OUT (SLIDER_WIDTH_IN * SCALE_FACTOR)
#define SLIDER_HEIGHT_OUT (SLIDER_HEIGHT_IN * SCALE_FACTOR)
#define NUM_SLIDERS_WIDTH (WIDTH_IN / SLIDER_WIDTH_IN)
#define NUM_SLIDERS_HEIGHT (HEIGHT_IN / SLIDER_HEIGHT_IN)
#define NUM_SLIDERS (WIDTH_IN * HEIGHT_IN / SLIDER_WIDTH_IN / SLIDER_HEIGHT_IN)

#define SLIDER_BUFFER_WIDTH (SLIDER_WIDTH_IN + 2)
#define SLIDER_BUFFER_HEIGHT (SLIDER_HEIGHT_IN + 2)
#define SLIDER_BUFFER_WIDTH_OUT (SLIDER_WIDTH_OUT + 4)
#define SLIDER_BUFFER_HEIGHT_OUT (SLIDER_HEIGHT_OUT + 4)


#define MARGIN_OF_ERROR 10


#endif
