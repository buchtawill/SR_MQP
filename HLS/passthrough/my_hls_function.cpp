#include "my_hls_function.h"
// HLS function to read from input stream and write to output stream

void my_hls_function(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return

	/*
	data_streamed input_data_stored[NUM_TRANSFERS];
	bool input_last_stored[NUM_TRANSFERS];
	ap_uint<BITS_PER_PIXEL / 8> input_keep_stored[NUM_TRANSFERS];
*/

	axis_t input_stored[NUM_TRANSFERS];


	int i = 0;
    // Process each input element
    while (!in_stream.empty()) {

    	//backup if more transfers than wanted are transferred
        if(i == NUM_TRANSFERS){
        	break;
        }

        #pragma HLS PIPELINE II=1

        // Read data from input stream
        input_stored[i] = in_stream.read();

        /*
        input_data_stored[i] = input_data.data;
        input_last_stored[i] = input_data.last;
        input_keep_stored[i] = input_data.keep; */

        i++;
    }

    //transfer values from array
    for(int j = 0; j < NUM_TRANSFERS; j++){
        axis_t output_data = input_stored[j];

        /*
        output_data.data = input_data_stored[j];
        output_data.last = input_last_stored[j];
        output_data.keep = input_keep_stored[j];
        output_data.strb = input_keep_stored[j]; */

        /*
        if(j == (NUM_TRANSFERS - 1)){
        	output_data.last = 1;
        }
        else{
        	output_data.last = 0;
        } */


        // Write data to output stream
        out_stream.write(output_data);
    }

    //reset array
    /*
    for(int j = 0; j < NUM_TRANSFERS; j++){
    	input_data_stored[j] = 0;
    } */
}
