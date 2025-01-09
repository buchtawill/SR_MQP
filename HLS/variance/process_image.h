#ifndef PROCESS_IMAGE_H
#define PROCESS_IMAGE_H

#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <ap_fixed.h>

// Define types for AXI4-Stream data
typedef ap_axiu<32, 1, 1, 1> axi_stream_in;
typedef ap_axiu<24, 1, 1, 1> axi_stream_out;

// Parameterization
const int IMAGE_WIDTH = 640; // Example width
const int IMAGE_HEIGHT = 480; // Example height
const int TILE_SIZE = 28; // Tile dimensions (28x28)

// Fixed-point type for internal calculations
typedef ap_fixed<16, 8> fixed_t;

// Function declaration
void process_image(hls::stream<axi_stream_in>& input_stream,
                   hls::stream<axi_stream_out>& output_stream);

#endif // PROCESS_IMAGE_H
