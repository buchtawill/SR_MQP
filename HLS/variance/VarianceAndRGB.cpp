#include "VarianceAndRGB.h"

// calculate variance on YUYV422
void process_tile(		hls::stream<axis_t> &pixel_stream_in,
        				hls::stream<axis_t> &conv_out,
						hls::stream<axis_t> &interp_out,
						unsigned int threshold,
						ap_uint<2> override_mode) {
#pragma HLS INTERFACE axis port=pixel_stream_in
#pragma HLS INTERFACE axis port=conv_out
#pragma HLS INTERFACE axis port=interp_out
#pragma HLS INTERFACE s_axilite port=threshold
#pragma HLS INTERFACE s_axilite port=override_mode
#pragma HLS INTERFACE s_axilite port=return

    unsigned int sum = 0;
    unsigned int variance;
    bool variance_calculated = false;

    data_stream pixel_data[PIXEL_COUNT];
	// #pragma HLS BIND_STORAGE variable=pixel_data type=RAM_1P impl=URAM

    unsigned int i = 0;
    while(i < PIXEL_COUNT / 2) {
//        #pragma HLS PIPELINE II=1
        while (!pixel_stream_in.empty()) {
        	if (i == PIXEL_COUNT / 2){
        		break;
        	}
            axis_t temp_input = pixel_stream_in.read();
            pixel_data[i] = temp_input.data;
            i++;
        }
    }

    if (override_mode == OVERRIDE_MODE_DEFAULT){
		for (i = 0; i < PIXEL_COUNT / 2; i++) {
			#pragma HLS PIPELINE II=15
			pixel_component Y0 = pixel_data[i](7, 0);
			pixel_component Y1 = pixel_data[i](23, 16);
			sum += static_cast<float>(Y0) + static_cast<float>(Y1);
		}

		unsigned int mean = sum / (PIXEL_COUNT);

		// calculate variance
		unsigned int variance_sum = 0;
		for (i = 0; i < PIXEL_COUNT / 2; i++) {
			#pragma HLS PIPELINE II=15
			pixel_component Y0 = pixel_data[i](7, 0);   // Lower 8 bits
			pixel_component Y1 = pixel_data[i](23, 16); // Next 8 bits

			float diff0 = static_cast<float>(Y0) - mean;
			float diff1 = static_cast<float>(Y1) - mean;

			variance_sum += (diff0 * diff0) + (diff1 * diff1);
		}

		variance = variance_sum / PIXEL_COUNT;
		variance_calculated = true;

		printf("Variance = %d \n", variance);
    }

    // YUYV422 --> RGB888 conversion here

	// logic for sending tiles
	for (i = 0; i < PIXEL_COUNT / 2; i++) {
		axis_t temp_output;
		temp_output.data = 0;
		temp_output.last = false;
		temp_output.keep = 0b1;
		temp_output.strb = 0b1;

		bool last;

		if(i == (PIXEL_COUNT / 2 - 1)){
			last = true;
		}
		else {
			last = false;
		}

		temp_output.data = pixel_data[i];
		temp_output.last = last;
		temp_output.keep = 0b1;
		temp_output.strb = 0b1;

		if (override_mode == OVERRIDE_MODE_CONV) { // send all tiles to convolution
			conv_out.write(temp_output);
		    interp_out.write({0, false, 0b0, 0b0});  // Ensure interp_out is empty

		} else if (override_mode == OVERRIDE_MODE_INTERP) {	// send all tiles to interpolation
			interp_out.write(temp_output);
		    conv_out.write({0, false, 0b0, 0b0});  // Ensure conv_out is empty

		}
		else if (variance_calculated){ // send based on variance
			if (variance > threshold) {
				conv_out.write(temp_output);
			    interp_out.write({0, false, 0b0, 0b0});  // Ensure interp_out is empty
			} else {
				interp_out.write(temp_output);
			    conv_out.write({0, false, 0b0, 0b0});  // Ensure conv_out is empty

			}
		}
	}
}
