#include "VarianceAndRGB.h"

// calculate variance on YUYV422
void process_tile(hls::stream<ap_uint<32>> &pixel_stream_in,
        				hls::stream<ap_uint<32>> &conv_out,
						hls::stream<ap_uint<32>> &interp_out,
						float threshold,
						ap_uint<2> override_mode) {

#pragma HLS INTERFACE axis port=pixel_stream_in
#pragma HLS INTERFACE axis port=conv_out
#pragma HLS INTERFACE axis port=interp_out
#pragma HLS INTERFACE s_axilite port=threshold
#pragma HLS INTERFACE s_axilite port=override_mode
#pragma HLS INTERFACE s_axilite port=return

    float luminance[PIXEL_COUNT];
    float sum = 0.0f;

    // read input, extract luminance (y component)
    for (int i = 0; i < PIXEL_COUNT / 2; i++) { // each 32 bits of YUYV is 2 pixels
		#pragma HLS PIPELINE II=1

        ap_uint<32> pixel_data = pixel_stream_in.read();

        // extract Y0 and Y1 from the 32-bit YUYV422 pixel
        ap_uint<8> Y0 = pixel_data(7, 0);  // Lower 8 bits
        ap_uint<8> U  = pixel_data(15, 8); // Next 8 bits (shared chroma)
        ap_uint<8> Y1 = pixel_data(23, 16); // Next 8 bits
        ap_uint<8> V  = pixel_data(31, 24); // Last 8 bits (shared chroma)

        // store y values and accumulate for mean calculation
        luminance[i*2] = static_cast<float>(Y0);
        luminance[(i*2) + 1] = static_cast<float>(Y1);
        sum += static_cast<float>(Y0) + static_cast<float>(Y1);
    }

    float mean = sum / PIXEL_COUNT;

    float variance_sum = 0.0f;
    for (int i = 0; i < PIXEL_COUNT; i++) {
		#pragma HLS PIPELINE II=1
        float diff = luminance[i] - mean;
        variance_sum += diff * diff;
    }

    float variance = variance_sum / PIXEL_COUNT;

    // logic for sending tiles
    for (int i = 0; i < PIXEL_COUNT / 2; i++) {
    	#pragma HLS PIPELINE II=1
            ap_uint<32> pixel_data = pixel_stream_in.read();

            if (override_mode == OVERRIDE_MODE_CONV) {
                // send all tiles to convolution
                conv_out.write(pixel_data);
            } else if (override_mode == OVERRIDE_MODE_INTERP) {
                // send all tiles to interpolation
                interp_out.write(pixel_data);
            } else {
                // send based on variance
                if (variance > threshold) {
                    conv_out.write(pixel_data);
                } else {
                    interp_out.write(pixel_data);
                }
            }
        }

    // YUYV422 --> RGB888 conversion here
}
