#include "bilinear_interpolation.h"
#include <ap_fixed.h>
#include <algorithm>

// HLS function for bilinear interpolation
int bilinear_interpolation_calculations(pixel_t image_in[HEIGHT_IN][WIDTH_IN],
                                        pixel_t image_out[HEIGHT_OUT][WIDTH_OUT]) {

	int temp_pixel_view;

    fixed widthRatio  = fixed(WIDTH_IN - 1) / fixed(WIDTH_OUT - 1);
    fixed heightRatio = fixed(HEIGHT_IN - 1) / fixed(HEIGHT_OUT - 1);

    for (int y_out = 0; y_out < HEIGHT_OUT; ++y_out) {

        #pragma HLS PIPELINE II=11

        for (int x_out = 0; x_out < WIDTH_OUT; ++x_out) {

            #pragma HLS UNROLL factor=2

            // Compute the corresponding input coordinates in fixed point
            fixed x_in = x_out * widthRatio;
            fixed y_in = y_out * heightRatio;

            // Determine the four nearest neighbors
            int x0 = static_cast<int>(x_in);
            int y0 = static_cast<int>(y_in);
            int x1 = std::min(x0 + 1, WIDTH_IN - 1);
            int y1 = std::min(y0 + 1, HEIGHT_IN - 1);

            // Calculate interpolation weights
            fixed dx = x_in - fixed(x0);
            fixed dy = y_in - fixed(y0);

            int show_dx, show_dy;
            show_dx = (int)dx;
            show_dy = (int)dy;

            fixed w00 = (fixed(1) - dx) * (fixed(1) - dy);
            fixed w10 = dx * (fixed(1) - dy);
            fixed w01 = (fixed(1) - dx) * dy;
            fixed w11 = dx * dy;

            int show_w00, show_w10, show_w01, show_w11;
            show_w00 = (int)w00;
            show_w10 = (int)w10;
            show_w01 = (int)w01;
            show_w11 = (int)w11;

            // Read pixel data from input (packed format 0xRRGGBB)
            pixel_t pixel00 = image_in[y0][x0];
            pixel_t pixel10 = image_in[y0][x1];
            pixel_t pixel01 = image_in[y1][x0];
            pixel_t pixel11 = image_in[y1][x1];

            int upper_left, upper_right, bottom_left, bottom_right;
            upper_left = (int)pixel00;
            upper_right = (int)pixel10;
            bottom_left = (int)pixel01;
            bottom_right = (int)pixel11;

            // Extract RGB channels
            channel_t b00 = (pixel00 >> 16) & 0xFF, g00 = (pixel00 >> 8) & 0xFF, r00 = pixel00 & 0xFF;
            channel_t b10 = (pixel10 >> 16) & 0xFF, g10 = (pixel10 >> 8) & 0xFF, r10 = pixel10 & 0xFF;
            channel_t b01 = (pixel01 >> 16) & 0xFF, g01 = (pixel01 >> 8) & 0xFF, r01 = pixel01 & 0xFF;
            channel_t b11 = (pixel11 >> 16) & 0xFF, g11 = (pixel11 >> 8) & 0xFF, r11 = pixel11 & 0xFF;

            // Compute interpolated values for each channel
            fixed r_interp = w00 * fixed(r00) + w10 * fixed(r10) + w01 * fixed(r01) + w11 * fixed(r11);
            fixed g_interp = w00 * fixed(g00) + w10 * fixed(g10) + w01 * fixed(g01) + w11 * fixed(g11);
            fixed b_interp = w00 * fixed(b00) + w10 * fixed(b10) + w01 * fixed(b01) + w11 * fixed(b11);

            int show_ri, show_gi, show_bi;
            show_ri = (int)r_interp;
            show_gi = (int)g_interp;
            show_bi = (int)b_interp;

            // Store interpolated values in `image_out` (rounded)
            pixel_t temp_pixel;
            temp_pixel.range(7, 0) = r_interp;
            temp_pixel.range(15, 8) = g_interp;
            temp_pixel.range(23, 16) = b_interp;

            temp_pixel_view = (int)temp_pixel;
            int temp_view_again = temp_pixel_view;

            image_out[y_out][x_out] = temp_pixel;
        }
    }

    return 1;
}

// Function to stream input samples into 2D array
void stream_samples_in(hls::stream<axis_t> &in_stream, pixel_t input_data_stored[HEIGHT_IN][WIDTH_IN]) {

    int i = 0;

    data_streamed input_streams_stored[NUM_TRANSFERS];

    while (i < NUM_TRANSFERS) {

        while (!in_stream.empty()) {

            if (i == NUM_TRANSFERS){
            	break;
            }

            axis_t temp_input = in_stream.read();
            input_streams_stored[i] = temp_input.data;

            i++;
        }
    }

    for (int transfer = 0; transfer < NUM_TRANSFERS; transfer++) {

        for (int j = 0; j < PIXELS_PER_TRANSFER; j++) {

            int index = transfer * PIXELS_PER_TRANSFER + j;
            int y = index / WIDTH_IN;
            int x = index % WIDTH_IN;
            input_data_stored[y][x] = input_streams_stored[transfer].range(j * BITS_PER_PIXEL + 23, j * BITS_PER_PIXEL);
        }
    }
}

// Function to stream output samples from 2D array
void stream_samples_out(pixel_t output_data_stored[HEIGHT_OUT][WIDTH_OUT], hls::stream<axis_t> &out_stream) {
    data_streamed loaded[NUM_TRANSFERS_OUT];

    for (int load = 0; load < NUM_TRANSFERS_OUT; load++) {

        data_streamed temp_load = 0;

        for (int pixel_transfer = 0; pixel_transfer < PIXELS_PER_TRANSFER; pixel_transfer++) {
            int index = load * PIXELS_PER_TRANSFER + pixel_transfer;
            int y = index / WIDTH_OUT;
            int x = index % WIDTH_OUT;

            temp_load.range(pixel_transfer * BITS_PER_PIXEL + 23, pixel_transfer * BITS_PER_PIXEL) = output_data_stored[y][x];
        }

        loaded[load] = temp_load;
    }

    for (int i = 0; i < NUM_TRANSFERS_OUT; i++) {
        axis_t output_stream;
        output_stream.data = loaded[i];
        output_stream.last = (i == NUM_TRANSFERS_OUT - 1);
        output_stream.keep = 0xFFFF;
        output_stream.strb = 0xFFFF;
        out_stream.write(output_stream);
    }
}

// Main function for bilinear interpolation processing
void bilinear_interpolation(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return

    pixel_t input_data_stored[HEIGHT_IN][WIDTH_IN];
    #pragma HLS BIND_STORAGE variable=input_data_stored type=RAM_2P impl=BRAM

    pixel_t output_data_stored[HEIGHT_OUT][WIDTH_OUT];
    #pragma HLS BIND_STORAGE variable=output_data_stored type=RAM_2P impl=BRAM

    #pragma HLS DATAFLOW
    stream_samples_in(in_stream, input_data_stored);

    bilinear_interpolation_calculations(input_data_stored, output_data_stored);

    stream_samples_out(output_data_stored, out_stream);
}
