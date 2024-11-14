#include <hls_stream.h>
#include <ap_int.h>
#include <cmath>
#include <iostream>

typedef ap_int<128> stream_data_t;
typedef ap_int<32> lite_data_t;

//void Interpolation_v2(hls::stream<stream_data_t> &image, hls::stream<stream_data_t> &featureMap, hls::stream<lite_data_t> &loadedInfo){
void Interpolation_v2(hls::stream<stream_data_t> &image, hls::stream<stream_data_t> &featureMap){
    #pragma HLS INTERFACE axis port=featureMap
    #pragma HLS INTERFACE axis port=image
    //#pragma HLS INTERFACE s_axilite port=loadedInfo
    #pragma HLS INTERFACE axi_cntrl_none port=return //allows Zynq/Microblaze to control IP core

    //store value from axi lite
    ap_uint<32> loadValues;
    //loadValues = loadedInfo.read();

    //mask and store values for image width and scaling factor when image width is the LSByte and scaling factor is next LSByte
    //0 in front indicates that its in hex
    //ap_uint<8> imageWidth = loadValues & 0xFF;
    //ap_uint<8> scalingFactor = loadValues & 0xFF00;
    
    //bus width: hoping to parameterize this
    const int bitsPerStream = 128;
    const int pixelsPerStream = bitsPerStream / 8;

    //input image widths: hoping to parameterize this
    const int imageWidth = 28;
    const int imageHeight = 28;
    const int upscalingFactor = 2;

    //dimensions of feature<Map
    int featureMapWidth = imageWidth * upscalingFactor;
    int featureMapHeight = imageHeight * upscalingFactor;

    //need to store image value streamed in so that it can be used for bilinear interpolation
    uint8_t imageStored[imageWidth*imageHeight*3];
    
    //need to store featureMap value to be streamed out
    //uint8_t featureMapStored[featureMapWidth * featureMapHeight];
    uint8_t featureMapStored[56 * 56*3];

    //consider storing in block ram instead of ultraram
    #pragma HLS bind_storage variable=imageStored core=XPM_MEMORY uram
    #pragma HLS bind_storage variable=featureMapStored core=XPM_MEMORY uram

    //store value from axi-stream
    ap_uint<128> imageLoadIn;
    //16, 8bit ints per 128 bit bus
    uint8_t imageLoadInArray[16];

    while(true){

        //147 reads to get full 28x28 image (with three channels per pixel)
        //can't do multiple reads to same data so should trigger new transfers each time, but not entirely sure
        for(int i = 0; i < 147; i++){
            //loads in image from axi-stream to array of uint_8
            imageLoadIn = image.read();
            for(int j = 0; j < pixelsPerStream; j++){
                //imageLoadInArray[j] = (imageLoadIn & (0xFF << (8 * ((pixelsPerStream - 1) - j)))) >> (8 * ((pixelsPerStream - 1) - j));
                imageStored[i*pixelsPerStream + j] = (imageLoadIn & (0xFF << (8 * ((pixelsPerStream - 1) - j)))) >> (8 * ((pixelsPerStream - 1) - j));
            }
        }

        //ADD STAGING GROUND HERE - would change image load in sequence

        // For each pixel in feature map image
        for (int fmY = 0; fmY < featureMapHeight; ++fmY){
            for (int fmX = 0; fmX < featureMapWidth; ++fmX)
            {
                // Map the pixel to the input image
                int x_in = static_cast<int>(std::round(fmX / upscalingFactor));
                int y_in = static_cast<int>(std::round(fmY / upscalingFactor));
                //float x_in = j / upscalingFactor;
                //float y_in = i / upscalingFactor;

                //location in input array of nearest pixels to one being interpolated 
                int x0 = static_cast<int>(std::floor(x_in));
                int x1 = std::min(x0 + 1, imageWidth - 1);
                int y0 = static_cast<int>(std::floor(y_in));
                int y1 = std::min(y0 + 1, imageHeight - 1);

                // Calculate the distances between the neighboring pixels -> should always be 1
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

                // per color (3 colors) per pixel
                for (int c = 0; c < 3; ++c)
                {
                    // Get the values of the four neighboring pixels
                    uint8_t p00 = imageStored[(y0 * imageWidth + x0) * 3 + c];
                    uint8_t p10 = imageStored[(y0 * imageWidth + x1) * 3 + c];
                    uint8_t p01 = imageStored[(y1 * imageWidth + x0) * 3 + c];
                    uint8_t p11 = imageStored[(y1 * imageWidth + x1) * 3 + c];

                    // Compute the interpolated pixel value
                    uint8_t interpolatedValue = static_cast<uint8_t>(std::round(w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11));

                    // Set the pixel value in the output image
                    featureMapStored[(fmY * featureMapWidth + fmX) * 3 + c] = interpolatedValue;
                } //end per channel
            }
        } //end bilinear interp calcs

        //588 transfers of 128 bits to pass out whole feature map
        for(int i = 0; i < 588; i++){

            //16 pixel color values per transfer
            for(int j = 0; j < pixelsPerStream; j++){
                ap_int<128> transValue = featureMapStored[i * pixelsPerStream + j] << (8 * ((pixelsPerStream - 1) - j));
            }

        }

    } //end while true
}
