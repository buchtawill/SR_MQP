#include "passthrough.h"
// HLS function to read from input stream and write to output stream

void passthrough(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return


    // Process each input element
    while (!in_stream.empty()) {

        #pragma HLS PIPELINE II=1
        // Read data from input stream

        axis_t input_data = in_stream.read();
        axis_t output_data;

        output_data.data = input_data.data;
        output_data.last = input_data.last;
        output_data.keep = input_data.keep;
        output_data.strb = input_data.keep;

        // Write data to output stream
        out_stream.write(output_data);
    }
}
