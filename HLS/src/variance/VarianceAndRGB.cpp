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
    fixed_32 mean;
    fixed_32 variance_sum;
    fixed_32 variance;
    bool variance_calculated = false;

    ap_uint_128 YUYV_pixel_data[YUYV_NUM_TRANSFERS];
    ap_uint_128 RGB_pixel_data[RGB_NUM_TRANSFERS];
	axis_t tmp_stream;
	tmp_stream.last = 0;

    unsigned int i = 0;
    while(!tmp_stream.last && i < YUYV_NUM_TRANSFERS){
		#pragma HLS PIPELINE II=1
		tmp_stream = pixel_stream_in.read();
		YUYV_pixel_data[i] = tmp_stream.data;
		i++;
	}

    // calculate mean
    if (override_mode == OVERRIDE_MODE_DEFAULT){
		for (i = 0; i < YUYV_NUM_TRANSFERS; i++) {
			#pragma HLS PIPELINE II=1
			for (int j = 0; j < 4; j++) {
				ap_8 Y0 = YUYV_pixel_data[i]((j * 32) + 7, (j * 32));
				ap_8 Y1 = YUYV_pixel_data[i]((j * 32) + 23, (j * 32) + 16);
				sum += Y0 + Y1;
			}
		}

		mean = sum / (PIXEL_COUNT);

		// calculate variance
		variance_sum = 0;
		for (i = 0; i < YUYV_NUM_TRANSFERS; i++) {
			#pragma HLS PIPELINE II=1
			for (int j = 0; j < 4; j++) {
				ap_8 Y0 = YUYV_pixel_data[i]((j * 32) + 7, (j * 32));
				ap_8 Y1 = YUYV_pixel_data[i]((j * 32) + 23, (j * 32) + 16);

				unsigned int diff0 = static_cast<fixed_32>(Y0) - mean;
				unsigned int diff1 = static_cast<fixed_32>(Y1) - mean;

				variance_sum += (diff0 * diff0) + (diff1 * diff1);
			}
		}

		variance = variance_sum / PIXEL_COUNT;
		variance_calculated = true;

		std::cout << "INFO: Variance: " << std::hex << (double)variance << std::endl;
    }

    // YUYV422 --> RGB888 conversion
    rgb_convert(RGB_pixel_data, YUYV_pixel_data);

	axis_t temp_output;
	temp_output.data = 0;


	/* 			output logic			*/

	// send all tiles to convolution (variance is not calculated)
    if (override_mode == OVERRIDE_MODE_CONV) {
    	for (i = 0; i < RGB_NUM_TRANSFERS; i++) {
			temp_output.last = (i == RGB_NUM_TRANSFERS - 1);
			temp_output.keep = 0xffff;
			temp_output.strb = 0xffff;
			temp_output.data = RGB_pixel_data[i];
			conv_out.write(temp_output);
    	}
	}

    // send all tiles to interpolation (variance is not calculated)
    else if (override_mode == OVERRIDE_MODE_INTERP) {
    	for (i = 0; i < RGB_NUM_TRANSFERS; i++) {
			temp_output.last = (i == RGB_NUM_TRANSFERS - 1);
			temp_output.keep = 0xffff;
			temp_output.strb = 0xffff;
			temp_output.data = RGB_pixel_data[i];
			interp_out.write(temp_output);
    	}
	}

    // send based on variance
    else if (variance_calculated) {
    	for (i = 0; i < RGB_NUM_TRANSFERS; i++) {
			temp_output.last = (i == RGB_NUM_TRANSFERS - 1);
			temp_output.keep = 0xffff;
			temp_output.strb = 0xffff;
			temp_output.data = RGB_pixel_data[i];

			if (variance > threshold) {
				conv_out.write(temp_output);
			}
			else {
				interp_out.write(temp_output);
			}
		}
    }
}


void rgb_convert(ap_uint_128 *RGB_pixel_data, ap_uint_128 *pixel_data) {
    int rgb_idx = 0;  // Output index

    for (int i = 0; i < YUYV_NUM_TRANSFERS; i++) {
		#pragma HLS PIPELINE II=1
        ap_uint_128 rgb_packed = 0;
        int bit_offset = 0;

        for (int j = 0; j < 8; j+= 2) {
            // extract YUYV values
        	ap_8 Y0 = pixel_data[i].range((j * 16) + 7, (j * 16));
        	ap_8 U  = pixel_data[i].range((j * 16) + 15, (j * 16) + 8);
        	ap_8 Y1 = pixel_data[i].range((j * 16) + 23, (j * 16) + 16);
        	ap_8 V  = pixel_data[i].range((j * 16) + 31, (j * 16) + 24);

//        	 constant integer calculations ////
        	// convert to RGB for first Y value
        	fixed_pixel R0 = (fixed_pixel)(Y0 + ((1436 * (V - 128) + 512) >> 10));
        	fixed_pixel G0 = (fixed_pixel)(Y0 - ((352 * (U - 128) + 512) >> 10) - ((732 * (V - 128) + 512) >> 10));
        	fixed_pixel B0 = (fixed_pixel)(Y0 + ((1811 * (U - 128) + 512) >> 10));

			// convert to RGB for second Y value
        	fixed_pixel R1 = (fixed_pixel)(Y1 + ((1436 * (V - 128) + 512) >> 10));
        	fixed_pixel G1 = (fixed_pixel)(Y1 - ((352 * (U - 128) + 512) >> 10) - ((732 * (V - 128) + 512) >> 10));
        	fixed_pixel B1 = (fixed_pixel)(Y1 + ((1811 * (U - 128) + 512) >> 10));
//        	///////////////////////////////////

        	// test print statements by pixel
//			std::cout << "INFO: R0: " << R0 << std::endl;
//			std::cout << "INFO: G0: " << G0 << std::endl;
//			std::cout << "INFO: B0: " << B0 << std::endl;
//			std::cout << "INFO: R1: " << R1 << std::endl;
//			std::cout << "INFO: G1: " << G1 << std::endl;
//			std::cout << "INFO: B1: " << B1 << std::endl;

//        	// float calculations /////////////
//        	fixed_pixel R0 = (fixed_pixel)(Y0 + 1.403 * (V - 128));
//        	fixed_pixel G0 = (fixed_pixel)(Y0 - 0.344 * (U - 128) - 0.714 * (V - 128));
//        	fixed_pixel B0 = (fixed_pixel)(Y0 + 1.770 * (U - 128));
//
//			fixed_pixel R1 = (fixed_pixel)(Y1 + 1.403 * (V - 128));
//			fixed_pixel G1 = (fixed_pixel)(Y1 - 0.344 * (U - 128) - 0.714 * (V - 128));
//			fixed_pixel B1 = (fixed_pixel)(Y1 + 1.770 * (U - 128));
//        	///////////////////////////////////

			std::cout << std::dec;

            // clamp to [0, 255]
            ap_8 R0_clamped = (R0 < RGB_MIN) ? RGB_MIN : ((R0 > RGB_MAX) ? RGB_MAX : R0);
            ap_8 G0_clamped = (G0 < RGB_MIN) ? RGB_MIN : ((G0 > RGB_MAX) ? RGB_MAX : G0);
            ap_8 B0_clamped = (B0 < RGB_MIN) ? RGB_MIN : ((B0 > RGB_MAX) ? RGB_MAX : B0);

            ap_8 R1_clamped = (R1 < RGB_MIN) ? RGB_MIN : ((R1 > RGB_MAX) ? RGB_MAX : R1);
            ap_8 G1_clamped = (G1 < RGB_MIN) ? RGB_MIN : ((G1 > RGB_MAX) ? RGB_MAX : G1);
            ap_8 B1_clamped = (B1 < RGB_MIN) ? RGB_MIN : ((B1 > RGB_MAX) ? RGB_MAX : B1);

            // test print statements by clamped pixel
//			std::cout << "INFO: R0 CLAMPED: " << R0_clamped << std::endl;
//			std::cout << "INFO: G0 CLAMPED: " << G0_clamped << std::endl;
//			std::cout << "INFO: B0 CLAMPED: " << B0_clamped << std::endl;
//			std::cout << "INFO: R1 CLAMPED: " << R1_clamped << std::endl;
//			std::cout << "INFO: G1 CLAMPED: " << G1_clamped << std::endl;
//			std::cout << "INFO: B1 CLAMPED: " << B1_clamped << std::endl;

            // pack as 0BGR (32-bit per pixel)
            ap_32 RGB0;
            RGB0.range(7,0) = R0_clamped;
            RGB0.range(15,8) = G0_clamped;
            RGB0.range(23,16) = B0_clamped;
            RGB0.range(31,24) = RGB_PAD;

            ap_32 RGB1;
            RGB1.range(7,0) = R1_clamped;
            RGB1.range(15,8) = G1_clamped;
            RGB1.range(23,16) = B1_clamped;
            RGB1.range(31,24) = RGB_PAD;

            rgb_packed.range(bit_offset + 31, bit_offset) = RGB0;
            rgb_packed.range(bit_offset + 63, bit_offset + 32) = RGB1;

            bit_offset += 64;

            if (bit_offset == 128) {
                RGB_pixel_data[rgb_idx++] = rgb_packed;
                rgb_packed = 0; // Reset for next set
                bit_offset = 0;
            }
        }
    }
}
