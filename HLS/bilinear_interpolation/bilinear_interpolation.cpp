#include "bilinear_interpolation.h"

//math for bilinear interpolation
void bilinear_interpolation_calculations(pixel_t image_in[HEIGHT_IN][WIDTH_IN][CHANNELS],
                           pixel_t image_out[HEIGHT_OUT][WIDTH_OUT][CHANNELS]) {

    // Perform Bilinear Interpolation
    for (int row_out = 0; row_out < HEIGHT_OUT; row_out++) {
        for (int col_out = 0; col_out < WIDTH_OUT; col_out++) {
            // Calculate corresponding position in the input image (floating point to allow interpolation)
            float row_in = row_out / (float)SCALE_FACTOR;
            float col_in = col_out / (float)SCALE_FACTOR;

            // Integer coordinates of the top-left pixel (floor values)
            int x1 = (int)row_in;
            int y1 = (int)col_in;

            // Fractional part of the coordinates (to calculate interpolation weights)
            float x_frac = row_in - x1;
            float y_frac = col_in - y1;

            // Clamping for boundary conditions (handling edges)
            int x2 = (x1 + 1) < HEIGHT_IN ? x1 + 1 : x1;
            int y2 = (y1 + 1) < WIDTH_IN ? y1 + 1 : y1;

            // Interpolation for each channel (e.g., RGB)
            for (int ch = 0; ch < CHANNELS; ch++) {
                // Interpolate horizontally first: between (x1, y1) and (x1, y2) for the top row,
                // and between (x2, y1) and (x2, y2) for the bottom row.
                float top_left = image_in[x1][y1][ch];
                float top_right = image_in[x1][y2][ch];
                float bottom_left = image_in[x2][y1][ch];
                float bottom_right = image_in[x2][y2][ch];

                // Perform bilinear interpolation in both horizontal and vertical directions
                float top_interp = (1 - y_frac) * top_left + y_frac * top_right; // Horizontal interpolation for top row
                float bottom_interp = (1 - y_frac) * bottom_left + y_frac * bottom_right; // Horizontal for bottom row

                // Now perform vertical interpolation between top_interp and bottom_interp
                float interpolated_value = (1 - x_frac) * top_interp + x_frac * bottom_interp;

                // Assign the interpolated value to the output image (clamping to pixel range)
                image_out[row_out][col_out][ch] = (pixel_t)(interpolated_value);
            }
        }
    }
}


void bilinear_interpolation(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return

    // Declare image buffers in URAM
    static pixel_t image_in[HEIGHT_IN][WIDTH_IN][CHANNELS];  // Input image stored in URAM
	#pragma HLS BIND_STORAGE variable=image_in type=RAM_1P impl=URAM

    // Temporary image buffer to store the output
    static pixel_t image_out[HEIGHT_OUT][WIDTH_OUT][CHANNELS]; // Output image buffer
	#pragma HLS BIND_STORAGE variable=image_out type=RAM_1P impl=URAM

    // Step 1: Read input image data from AXI-Stream to image_in
    while(!in_stream.empty()){
		for (int row = 0; row < HEIGHT_IN; row++) {
			for (int col = 0; col < WIDTH_IN; col++) {
				for (int ch = 0; ch < CHANNELS; ch++) {
					#pragma HLS PIPELINE II=1
					axis_t input_data = in_stream.read();
					image_in[row][col][ch] = input_data.data;
				}
			}
		}
    }


    // Step 2: Perform bilinear interpolation for upscaling the image
    bilinear_interpolation_calculations(image_in, image_out); // Pass image_out by reference

    // Step 3: Write the output image to AXI-Stream
    for (int row_out = 0; row_out < HEIGHT_OUT; row_out++) {
        for (int col_out = 0; col_out < WIDTH_OUT; col_out++) {
            for (int ch = 0; ch < CHANNELS; ch++) {
                axis_t output_data;
                output_data.data = image_out[row_out][col_out][ch];
                output_data.last = (row_out == HEIGHT_OUT - 1 && col_out == WIDTH_OUT - 1 && ch == CHANNELS - 1);
                out_stream.write(output_data);
            }
        }
    }
}
