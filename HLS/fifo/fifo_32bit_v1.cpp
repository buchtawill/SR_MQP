#include "fifo_32bit_v1.h"
// HLS function to read from input stream and write to output stream

void fifo_32bit_v1(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return

	data_streamed input_data_stored[NUM_TRANSFERS];
	#pragma HLS BIND_STORAGE variable=input_data_stored type=RAM_1P impl=URAM

	bool input_last_stored[NUM_TRANSFERS];
	#pragma HLS BIND_STORAGE variable=input_last_stored type=RAM_1P impl=URAM

	ap_uint<BITS_PER_TRANSFER / 8> input_keep_stored[NUM_TRANSFERS];
	#pragma HLS BIND_STORAGE variable=input_keep_stored type=RAM_1P impl=URAM

	//axis_t input_stored[NUM_TRANSFERS];


	int i = 0;

	//update later with reset signal, but make sure FIFO is cleared on start up
	if(i == NUM_TRANSFERS || i == 0){
		for(int k = 0; k < NUM_TRANSFERS; k++){
			input_data_stored[k] = 0;
			input_last_stored[k] = 0;
			input_keep_stored[k] = 0;
		}
	}


	//make sure the correct number of transfers are passed in
	while(i < NUM_TRANSFERS){

		while(!in_stream.empty()){

			//if the correct number of transfers have been received stop taking in new data
			if(i == NUM_TRANSFERS){
				break;
			}

			axis_t temp_input = in_stream.read();
	        input_data_stored[i] = temp_input.data;
	        input_last_stored[i] = temp_input.last;
	        input_keep_stored[i] = temp_input.keep;
			i++;
		}
	}



	int j = 0;
    //once all the data has been read in
	//this might need to be NUM_TRANSFERS - 1
    if(i >= NUM_TRANSFERS){

        //transfer values from array
        while(j < NUM_TRANSFERS){

            //axis_t output_data = input_stored[j];
        	axis_t temp_output;


            //ap_uint<BITS_PER_TRANSFER / 8> keep;
            bool last;

            /*
            if(j % 28 == 0){
            	keep = 1;
            }
            else{
            	keep = 0;
            } */

            if(j == (NUM_TRANSFERS - 1)){
            	last = true;
            }
            else {
            	last = false;
            }

        	//don't change any of the signals that were passed in
            temp_output.data = input_data_stored[j];
            temp_output.last = last;
            temp_output.keep = 0xF;
            temp_output.strb = 0xF;

            //temp_output.last = input_last_stored[j];
            //temp_output.keep = input_keep_stored[j];
            //temp_output.strb = input_keep_stored[j];

            /*
            if(j == (NUM_TRANSFERS - 1)){
            	output_data.last = 1;
            }
            else{
            	output_data.last = 0;
            } */


            // Write data to output stream
            out_stream.write(temp_output);

            j++;
        }
    }


    //reset array and allow new transfers
    if(i >= NUM_TRANSFERS && j >= NUM_TRANSFERS){

		for(int k = 0; k < NUM_TRANSFERS; k++){
			input_data_stored[k] = 0;
			input_last_stored[k] = 0;
			input_keep_stored[k] = 0;
		}

		i = 0;
		j = 0;
    }
}
