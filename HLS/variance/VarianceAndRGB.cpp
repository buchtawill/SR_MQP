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

    axis_t pixel_data[PIXEL_COUNT];
	// #pragma HLS BIND_STORAGE variable=pixel_data type=RAM_1P impl=URAM

    unsigned int i = 0;
    while(i < PIXEL_COUNT / 2) {
        #pragma HLS PIPELINE II=6
    	axis_t temp_input;
        if (!pixel_stream_in.empty()) {
            temp_input = pixel_stream_in.read();
            pixel_data[i] = temp_input;
            pixel_component Y0 = pixel_data[i].data(7, 0);
            pixel_component Y1 = pixel_data[i].data(23, 16);

            sum += static_cast<float>(Y0) + static_cast<float>(Y1);
            i++;
        }
    }

    /*
    // read input
    while (i < PIXEL_COUNT / 2) { // each 32 bits of YUYV is 2 pixels
		#pragma HLS PIPELINE II=3
    	if (!pixel_stream_in.empty()) {
			pixel_data[i] = pixel_stream_in.read();
			i++;
    	}
    }
    */

    /*
    for (int i = 0; i < PIXEL_COUNT / 2; i++) {
		#pragma HLS PIPELINE II=3
    	// extract Y0 and Y1 from the 32-bit YUYV422 pixel
		pixel_component Y0 = pixel_data[i](7, 0);  // Lower 8 bits
		pixel_component U  = pixel_data[i](15, 8); // Next 8 bits (shared chroma)
		pixel_component Y1 = pixel_data[i](23, 16); // Next 8 bits
		pixel_component V  = pixel_data[i](31, 24); // Last 8 bits (shared chroma)

		// store y values and accumulate for mean calculation
		luminance[i*2] = static_cast<float>(Y0);
		luminance[(i*2) + 1] = static_cast<float>(Y1);
		sum += static_cast<float>(Y0) + static_cast<float>(Y1);
    }

        float variance_sum = 0.0f;
    for (int i = 0; i < PIXEL_COUNT / 2; i++) {
		#pragma HLS PIPELINE II=3
        float diff = luminance[i] - mean;
        variance_sum += diff * diff;
    }
    */

//    for (int i = 0; i < PIXEL_COUNT / 2; i++) {
//            #pragma HLS PIPELINE II=3
//            pixel_component Y0 = pixel_data[i](7, 0);
//            pixel_component Y1 = pixel_data[i](23, 16);
//
//            sum += static_cast<float>(Y0) + static_cast<float>(Y1);
//        }

    unsigned int mean = sum / (PIXEL_COUNT);

    // calculate variance
	unsigned int variance_sum = 0;
	for (int i = 0; i < PIXEL_COUNT / 2; i++) {
		#pragma HLS PIPELINE II=6
		pixel_component Y0 = pixel_data[i].data(7, 0);   // Lower 8 bits
		pixel_component Y1 = pixel_data[i].data(23, 16); // Next 8 bits

		float diff0 = static_cast<float>(Y0) - mean;
		float diff1 = static_cast<float>(Y1) - mean;

		variance_sum += (diff0 * diff0) + (diff1 * diff1);
	}

    unsigned int variance = variance_sum / PIXEL_COUNT;

     printf("Variance = %d \n", variance);

    // logic for sending tiles
    for (int i = 0; i < PIXEL_COUNT / 2; i++) {
		if (override_mode == OVERRIDE_MODE_CONV) { // send all tiles to convolution
			conv_out.write(pixel_data[i]);
		} else if (override_mode == OVERRIDE_MODE_INTERP) {	// send all tiles to interpolation
			interp_out.write(pixel_data[i]);
		}
		else { // send based on variance
			if (variance > threshold) {
				conv_out.write(pixel_data[i]);
			} else {
				interp_out.write(pixel_data[i]);
			}
		}
	}

    // YUYV422 --> RGB888 conversion here
}

#include "VarianceAndRGB.h"

// calculate variance on YUYV422
void process_tile(		hls::stream<data_stream> &pixel_stream_in,
        				hls::stream<data_stream> &conv_out,
						hls::stream<data_stream> &interp_out,
						float threshold,
						ap_uint<2> override_mode) {
#pragma HLS INTERFACE axis port=pixel_stream_in
#pragma HLS INTERFACE axis port=conv_out
#pragma HLS INTERFACE axis port=interp_out
#pragma HLS INTERFACE s_axilite port=threshold
#pragma HLS INTERFACE s_axilite port=override_mode
#pragma HLS INTERFACE s_axilite port=return

    float luminance[PIXEL_COUNT]; // store this in URAM
    float sum = 0.0f;
    data_stream pixel_data[PIXEL_COUNT];
    unsigned int i = 0;

    // read input, extract luminance (y component)
    while (!pixel_stream_in.empty() && (i < PIXEL_COUNT / 2)) { // each 32 bits of YUYV is 2 pixels
//		#pragma HLS PIPELINE II=1

    	if (!pixel_stream_in.empty()) {
			pixel_data[i] = pixel_stream_in.read();

			// extract Y0 and Y1 from the 32-bit YUYV422 pixel
			pixel_component Y0 = pixel_data[i](7, 0);  // Lower 8 bits
			pixel_component U  = pixel_data[i](15, 8); // Next 8 bits (shared chroma)
			pixel_component Y1 = pixel_data[i](23, 16); // Next 8 bits
			pixel_component V  = pixel_data[i](31, 24); // Last 8 bits (shared chroma)

			// store y values and accumulate for mean calculation
			luminance[i*2] = static_cast<float>(Y0);
			luminance[(i*2) + 1] = static_cast<float>(Y1);
			sum += static_cast<float>(Y0) + static_cast<float>(Y1);
			i++;
    	}
    }

    float mean = sum / PIXEL_COUNT;

    float variance_sum = 0.0f;
    for (int i = 0; i < PIXEL_COUNT / 2; i++) {
//		#pragma HLS PIPELINE II=1
        float diff = luminance[i] - mean;
        variance_sum += diff * diff;
    }
    int variance = variance_sum / PIXEL_COUNT;

    // printf("Variance = %d \n", variance);

    // logic for sending tiles
    for (int i = 0; i < PIXEL_COUNT / 2; i++) {
//				#pragma HLS PIPELINE II=1
//            	data_stream pixel_data = pixel_stream_in.read();


		if (override_mode == OVERRIDE_MODE_CONV) {
			// send all tiles to convolution
			conv_out.write(pixel_data[i]);
		} else if (override_mode == OVERRIDE_MODE_INTERP) {
			// send all tiles to interpolation
			interp_out.write(pixel_data[i]);
		} else {
			// send based on variance
			if (variance > threshold) {
				conv_out.write(pixel_data[i]);
			} else {
				interp_out.write(pixel_data[i]);
			}
		}
	}

    // YUYV422 --> RGB888 conversion here
}
