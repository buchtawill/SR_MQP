#include "fifo.h"

void fifo(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return

    // Declare image buffers in URAM
    static pixel_t image_in[HEIGHT_IN][WIDTH_IN][CHANNELS];  // Input image stored in URAM
	#pragma HLS BIND_STORAGE variable=image_in type=RAM_1P impl=URAM

    // Temporary image buffer to store the output
    static pixel_t image_out[HEIGHT_OUT][WIDTH_OUT][CHANNELS]; // Output image buffer
	#pragma HLS BIND_STORAGE variable=image_out type=RAM_1P impl=URAM

    // Step 1: Read input image data from AXI-Stream to image_in
    while(!in_stream.empty()){
		for (int row = 0; row < HEIGHT_IN; row++) {
			for (int col = 0; col < WIDTH_IN; col++) {
				for (int ch = 0; ch < CHANNELS; ch++) {
					#pragma HLS PIPELINE II=1
					axis_t input_data = in_stream.read();
					image_in[row][col][ch] = input_data.data;
				}
			}
		}
    }

    // Step 3: Write the output image to AXI-Stream
    for (int row_out = 0; row_out < HEIGHT_OUT; row_out++) {
        for (int col_out = 0; col_out < WIDTH_OUT; col_out++) {
            for (int ch = 0; ch < CHANNELS; ch++) {
                axis_t output_data;
                output_data.data = image_out[row_out][col_out][ch];
                output_data.last = (row_out == HEIGHT_OUT - 1 && col_out == WIDTH_OUT - 1 && ch == CHANNELS - 1);
                out_stream.write(output_data);
            }
        }
    }
}
