#include <hls_stream.h>
#include <ap_int.h>
#include <cmath>
#include <iostream>

typedef ap_int<128> stream_data_t;
typedef ap_int<32> lite_data_t;

void Interpolation_v2(hls::stream<stream_data_t> &image, hls::stream<stream_data_t> &featureMap, hls::stream<lite_data_t> &loadedInfo){
    #pragma HLS INTERFACE axis port=featureMap
    #pragma HLS INTERFACE axis port=image
    #pragma HLS INTERFACE s_axilite port=loadedInfo
    #pragma HLS INTERFACE axi_cntrl_none port=return //allows Zynq/Microblaze to control IP core

    //bus width: hoping to parameterize this
    const int bitsPerStream = 128;

    //input image widths: hoping to parameterize this
    const int inputWidth = 28;
    const int inputHeight = 28;
    const int upscalingFactor = 2;

    //store value from axi lite
    ap_uint<8> loadValues[32];
    //loadValues = loadedInfo.read();

    //mask and store values for image width and scaling factor when image width is the LSByte and scaling factor is next LSByte
    //0 in front indicates that its in hex
    //ap_uint<8> imageWidth = loadValues & 0xFF;
    //ap_uint<8> scalingFactor = loadValues & 0xFF00;

    //dimensions of feature<Map
    int outputWidth = inputWidth * scale;
    int outputHeight = inputHeight * scale;

    // For each pixel in the output image
    for (int i = 0; i < outputHeight; ++i){
        for (int j = 0; j < outputWidth; ++j)
        {
            // Map the pixel to the input image
            int x_in = static_cast<int>(std::round(j / scale));
            int y_in = static_cast<int>(std::round(i / scale));
            //float x_in = x_out / scale;
            //float y_in = y_out / scale;

            //location in input array of nearest pixels to one being interpolated 
            int x0 = static_cast<int>(std::floor(x_in));
            int x1 = std::min(x0 + 1, width - 1);
            int y0 = static_cast<int>(std::floor(y_in));
            int y1 = std::min(y0 + 1, height - 1);

            // Calculate the distances between the neighboring pixels
            int dx = static_cast<int>(std::round(x_in - x0));
            int dy = static_cast<int>(std::round(y_in - y0));
            //float dx = x_in - x0;
            //float dy = y_in - y0;

            // Compute interpolation weights
            int w00 = static_cast<int>(std::round((1 - dx) * (1 - dy)));
            int w10 = static_cast<int>(std::round(dx * (1 - dy)));
            int w01 = static_cast<int>(std::round((1 - dx) * dy));
            int w11 = static_cast<int>(std::round(dx * dy));
            //float w00 = (1 - dx) * (1 - dy);
            //float w10 = dx * (1 - dy);
            //float w01 = (1 - dx) * dy;
            //float w11 = dx * dy;

            // per color per pixel
            for (int c = 0; c < 3; ++c)
            {
                // Get the values of the four neighboring pixels
                //uint8 because 0-255
                uint8_t p00 = image[(y0 * inputWidth + x0) * 3 + c];
                uint8_t p10 = image[(y0 * inputWidth + x1) * 3 + c];
                uint8_t p01 = image[(y1 * inputWidth + x0) * 3 + c];
                uint8_t p11 = image[(y1 * inputWidth + x1) * 3 + c];

                // Compute the interpolated pixel value
                uint8_t interpolatedValue = static_cast<uint8_t>(std::round(w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11));

                // Set the pixel value in the output image
                outputImage[(i * outputWidth + j) * 3 + c] = interpolatedValue;
            }
        }
    }