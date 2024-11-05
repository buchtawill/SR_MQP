//throwing errors in VS Code, but I think thats because its not referencing what was downloaded in Vitis HLS
#include <hls_stream.h>
#include <ap_int.h>
#include <cmath>



//based off of this https://gist.github.com/folkertdev/6b930c7a7856e36dcad0a72a03e66716
//assumes that image and feature map will always be squares, otherwise would need to take in imageHeight
void interp_top(hls::stream<ap_int<1024>> &image, hls::stream<ap_int<1024>> &featureMap, hls::stream<ap_int<32>> &imageWidth, hls::stream<ap_int<32>> &scalingFactor){
    #pragma HLS INTERFACE axis port=featureMap
    #pragma HLS INTERFACE axis port=image
    #pragma HLS INTERFACE s_axilite port=imageWidth
    #pragma HLS INTERFACE s_axilite port=scalingFactor
    #pragma HLS INTERFACE axi_cntrl_none port=return //allows Zynq/Microblaze to control IP core

    //parameterize
    /*
    const int imageWidth = 28;
    const int imageHeight = 28;
    const int featureMapWidth = 56;
    const int featureMapHeight = 56;  
    */  
       

    //need to store image value streamed in so that it can be used for bilinear interpolation 
    int imageStored[imageWidth*imageWidth];
    int featureMapStored[imageWidth*scalingFactor*imageWidth*scalingFactor];

    int imageSize = imageWidth*imageWidth;
    int featureMapWidth = imageWidth*scalingFactor;

    while(true){

        //read in and store the image values
        for(int i = 0; i < imageSize; i++){
            imageStored[i] = image.read();
        }

        //integer intervals: determines the location of the interpolated points
        float ratio = (imageWidth - 1) / (featureMapWidth - 1);
        //float y_ratio = (imageHeight - 1) / (featureMapHeight - 1);

        //iterate through each of the output rows
        for(int i = 0; i < featureMapWidth; i++){
            //iterate through each of the output columns
            for(int j = 0; j < featureMapWidth; j++){

                //determies x values for the coordinates to the left and right of the pixel being interpolated
                int x_l = floor(ratio * (float)j);
                int x_h = ceil(ratio * (float)j);

                //determines the y values for the coordinate to the top and bottom of the pixel being interpolated
                int y_l = floor(ratio * (float)i);
                int y_h = ceil(ratio * (float)i);

                //might not need weights since they are all the same distance from the other points (1 away)
                //float x_weight = (x_ratio * (float)j) - x_l;
                //float y_weight = (y_ratio * (float)i) - y_l;

                //gets the values from the four pixels you are interpolating from
                float a = imageStored[y_l * imageWidth + x_l];
                float b = imageStored[y_l * imageWidth + x_h];
                float c = imageStored[y_h * imageWidth + x_l];
                float d = imageStored[y_h * imageWidth + x_h];

                //don't need to multiply by weights since they are all equadistant
                int pixel = int((a + b + c + d)/4);

                featureMapStored[i * featureMapWidth + j] = pixel;

            }
        }

        //pass interpolated feature map out through stream
        for(int i = 0; i < featureMapWidth * featureMapWidth; i++){
            featureMap.write(featureMapStored[i]);
        }

    } //while true

}