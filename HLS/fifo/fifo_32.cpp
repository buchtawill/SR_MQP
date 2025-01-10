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

    // Step 1: Read input values from the stream and store them in the buffer
    int row = 0, col = 0, ch = 0;

    while (!in_stream.empty()) {
        #pragma HLS PIPELINE II=1
        axis_t input_data = in_stream.read(); // Read data from the stream

        // Store the data in the buffer
        image_in[row][col][ch] = input_data.data;

        // Increment indices
        ch++;
        if (ch == CHANNELS) {
            ch = 0;
            col++;
            if (col == WIDTH_IN) {
                col = 0;
                row++;
                if (row == HEIGHT_IN) {
                    break; // Stop reading when the buffer is filled
                }
            }
        }
    }

    // Step 2: Write the values from the buffer back to the stream
    for (int row_out = 0; row_out < HEIGHT_IN; row_out++) {
        for (int col_out = 0; col_out < WIDTH_IN; col_out++) {
            for (int ch_out = 0; ch_out < CHANNELS; ch_out++) {
                #pragma HLS PIPELINE II=1
                axis_t output_data;
                output_data.data = image_in[row_out][col_out][ch_out]; // Read data from the buffer
                output_data.last = (row_out == HEIGHT_IN - 1 && col_out == WIDTH_IN - 1 && ch_out == CHANNELS - 1); // Indicate last value
                out_stream.write(output_data); // Write data to the output stream
            }
        }
    }
}
