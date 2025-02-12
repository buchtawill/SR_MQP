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
    fixed mean;
    fixed variance_sum;
    fixed variance;
    bool variance_calculated = false;

    data_stream pixel_data[NUM_TRANSFERS];
	axis_t tmp_stream;
	tmp_stream.last = 0;
	// #pragma HLS BIND_STORAGE variable=pixel_data type=RAM_1P impl=URAM

    unsigned int i = 0;
    while(!tmp_stream.last && i < NUM_TRANSFERS){
		#pragma HLS PIPELINE II=1
		tmp_stream = pixel_stream_in.read();
		pixel_data[i] = tmp_stream.data;
		i++;
	}

    if (override_mode == OVERRIDE_MODE_DEFAULT){
		for (i = 0; i < NUM_TRANSFERS; i++) {
			#pragma HLS PIPELINE II=1
			for (int j = 0; j < 4; j++) {
				ap_8 Y0 = pixel_data[i]((j * 32) + 7, (j * 32));
				ap_8 Y1 = pixel_data[i]((j * 32) + 23, (j * 32) + 16);
				sum += Y0 + Y1;
			}
		}

		mean = sum / (PIXEL_COUNT);

		// calculate variance
		variance_sum = 0;
		for (i = 0; i < NUM_TRANSFERS; i++) {
			#pragma HLS PIPELINE II=1
			for (int j = 0; j < 4; j++) {
				ap_8 Y0 = pixel_data[i]((j * 32) + 7, (j * 32));
				ap_8 Y1 = pixel_data[i]((j * 32) + 23, (j * 32) + 16);

				unsigned int diff0 = static_cast<fixed>(Y0) - mean;
				unsigned int diff1 = static_cast<fixed>(Y1) - mean;

				variance_sum += (diff0 * diff0) + (diff1 * diff1);
			}
		}

		variance = variance_sum / PIXEL_COUNT;
		variance_calculated = true;

		std::cout << "INFO: Variance: " << std::hex << (double)variance << std::endl;
    }

    // YUYV422 --> RGB888 conversion here

	// logic for sending tiles
	for (i = 0; i < NUM_TRANSFERS; i++) {
		axis_t temp_output;
		temp_output.data = 0;
		temp_output.last = false;
		temp_output.keep = 0b1;
		temp_output.strb = 0b1;

		bool last;

		if(i == (NUM_TRANSFERS - 1)){
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
