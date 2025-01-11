#include "my_hls_function.h"
// HLS function to read from input stream and write to output stream

void my_hls_function(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return

	axis_t input_stored[NUM_TRANSFERS];

	int i = 0;
    // Process each input element
    while (!in_stream.empty()) {

        if(i == NUM_TRANSFERS){
        	break;
        }

        #pragma HLS PIPELINE II=1
        // Read data from input stream
        axis_t input_data = in_stream.read();
        input_stored[i] = input_data;

        i++;
    }

    for(int j = 0; j < NUM_TRANSFERS; j++){
        axis_t output_data = input_stored[j];

        /*
        output_data.data = input_data.data;
        output_data.last = input_data.last;
        output_data.keep = 1;
        output_data.strb = 1; */

        // Write data to output stream
        out_stream.write(output_data);
    }
}
