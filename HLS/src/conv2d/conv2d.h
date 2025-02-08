#ifndef CONV2D_H
#define CONV2D_H

#include <ap_axi_sdata.h>
#include "hls_stream.h"
#include <ap_int.h>

#define STREAM_WIDTH            128
#define INPUT_WIDTH             28
#define INPUT_HEIGHT            28
#define BYTES_PER_PIXEL         3
#define NUM_TRANSFER_BYTES      (INPUT_WIDTH * INPUT_HEIGHT * BYTES_PER_PIXEL)
#define BYTES_PER_TRANSFER      (STREAM_WIDTH / 8)
#define STREAM_BEATS_PER_TILE   ((NUM_TRANSFER_BYTES * 8) / STREAM_WIDTH)

typedef ap_uint<STREAM_WIDTH> stream_data_t;

// Define axis_t with data width of 8 bits and no additional signals
typedef hls::axis<stream_data_t, 0, 0, 0> axis_t;

void conv2d_top(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream);

#endif // CONV2D_H
