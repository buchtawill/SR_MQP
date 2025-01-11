#include "my_hls_function.h"

// HLS function to read from input stream and write to output stream
void my_hls_function(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return

	axis_t data_stored[BYTES_TRANSFERRED];

	int i = 0;
    // Process each input element
    while (!in_stream.empty()) {
        #pragma HLS PIPELINE II=1

        // Read data from input stream
        axis_t input_data = in_stream.read();
        if(i < 16){
        	data_stored[i] = input_data;
        }

        i++;
    }


    for(int j = 0; j < BYTES_TRANSFERRED; j++){
    	out_stream.write(data_stored[j]);
    }
}
