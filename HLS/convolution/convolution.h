#include <hls_stream.h>
#include <ap_int.h>
#include <iostream>

// Define the matrix dimensions
#define MATRIX_SIZE 28    // Example size of the input/output matrix (8x8)
#define KERNEL_SIZE 3    // Example kernel size (3x3)
#define NUM_CHANNELS 3
// Define the pixel type as 8-bit unsigned integer
typedef ap_uint<8> pixel_t;

// Define the AXI-Stream interface for input/output matrices
struct axi_stream {
    pixel_t data;
    bool last;
};

//Function declarations
void convolution_top(hls::stream<axi_stream> &in_stream, hls::stream<axi_stream> &out_stream, hls::stream<axi_stream> &kernel_stream);
