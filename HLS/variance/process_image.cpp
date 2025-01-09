#include "process_image.h"

void process_image(hls::stream<axi_stream_in>& input_stream,
                   hls::stream<axi_stream_out>& output_stream,
                   int tile_width, int tile_height) {
    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream
    #pragma HLS INTERFACE s_axilite port=tile_width
    #pragma HLS INTERFACE s_axilite port=tile_height
    #pragma HLS INTERFACE s_axilite port=return

    // Tile buffer for storing Y channel dynamically
    static fixed_t tile_y[1024][1024]; // Maximum tile size limits
    #pragma HLS RESOURCE variable=tile_y core=RAM_2P_BRAM

    // Iterate over tiles
    for (int tile_y_idx = 0; tile_y_idx < IMAGE_HEIGHT; tile_y_idx += tile_height) {
        for (int tile_x_idx = 0; tile_x_idx < IMAGE_WIDTH; tile_x_idx += tile_width) {
            #pragma HLS PIPELINE II=1

            // Variables for mean and variance calculation
            fixed_t sum_y = 0;
            fixed_t sum_sq_y = 0;

            // Read tile data and calculate sum and sum of squares
            for (int i = 0; i < tile_height; ++i) {
                for (int j = 0; j < tile_width; ++j) {
                    axi_stream_in in_data = input_stream.read();
                    ap_uint<32> pixel = in_data.data;

                    // Extract Y channel from YUYV422
                    ap_uint<8> y = (j % 2 == 0) ? pixel.range(7, 0) : pixel.range(23, 16);
                    fixed_t y_fixed = (fixed_t)y;
                    tile_y[i][j] = y_fixed;
                    sum_y += y_fixed;
                    sum_sq_y += y_fixed * y_fixed;
                }
            }

            // Calculate mean and variance
            fixed_t num_pixels = tile_width * tile_height;
            fixed_t mean_y = sum_y / num_pixels;
            fixed_t variance_y = (sum_sq_y / num_pixels) - (mean_y * mean_y);

            // Convert and write RGB888 data for the tile
            for (int i = 0; i < tile_height; ++i) {
                for (int j = 0; j < tile_width; ++j) {
                    axi_stream_out out_data;
                    axi_stream_in in_data = input_stream.read(); // Read the stream
                    ap_uint<32> pixel = in_data.data; // Extract the 32-bit pixel data
                    ap_uint<8> y = (j % 2 == 0) ? pixel.range(7, 0) : pixel.range(23, 16);
                    ap_uint<8> u = pixel.range(15, 8);
                    ap_uint<8> v = pixel.range(31, 24);

                    // Fixed-point Color space conversion (YUYV422 to RGB888)
                    fixed_t r = y + 1.402 * (v - 128);
                    fixed_t g = y - 0.344136 * (u - 128) - 0.714136 * (v - 128);
                    fixed_t b = y + 1.772 * (u - 128);

                    // Clamp values to [0, 255]
                    ap_uint<8> r_clamped = (ap_uint<8>)((r < (fixed_t)0) ? (fixed_t)0 : ((r > (fixed_t)255) ? (fixed_t)255 : r));
                    ap_uint<8> g_clamped = (ap_uint<8>)((g < (fixed_t)0) ? (fixed_t)0 : ((g > (fixed_t)255) ? (fixed_t)255 : g));
                    ap_uint<8> b_clamped = (ap_uint<8>)((b < (fixed_t)0) ? (fixed_t)0 : ((b > (fixed_t)255) ? (fixed_t)255 : b));

                    // Pack RGB888
                    out_data.data = (r_clamped, g_clamped, b_clamped);
                    output_stream.write(out_data);
                }
            }
        }
    }
}
