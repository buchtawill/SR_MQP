//throwing errors in VS Code, but I think thats because its not referencing what was downloaded in Vitis HLS
//#include "hls_math.h"
//#include "hlslib.h"
#include “ap_int.h”
//TAMD documentation didn't mention these libraries, but they were used in various tutorials I found
//#include <iostream>
//#include <cmath>
//#include <hls_stream.h>
//#include "ap_axi_sdata.h"

//8-bit interger with side-channel, used for the TLAST signal which indicates streaming is done
typedef ap_axis<8, 2, 5, 6> intSdCh;

//based off of this https://gist.github.com/folkertdev/6b930c7a7856e36dcad0a72a03e66716
void interp_top(hls::stream<int8_t> &image, hls::stream<intSdCh> &featureMap){
    #pragma HLS INTERFACE axis port=featureMap
    #pragma HLS INTERFACE axi port=image
    #pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS //allows Zynq/Microblaze to control IP core

    //parameterize
    const int imageWidth = 28;
    const int imageHeight = 28;
    const int featureMapWidth = 56;
    const int featureMapHeight = 56;   

    //need to store image value streamed in so that it can be used for bilinear interpolation 
    int8 imageStored[imageWidth*imageHeight];
    intSdCh featureMapStored[featureMapWidth*featureMapHeight];

    for(int i = 0; i < imageWidth * imageHeight; i++){
    //#pragma HLS PIPELINE
        imageStored = image.read();
    }

    //generate interpolated feature map

    float x_ratio = (imageWidth - 1) / (featureMapWidth - 1);
    float y_ratio = (imageHeight - 1) / (featureMapHeight - 1);

    //create output values
    //iterate through each of the output rows
    for(int i = 0; i < featureMapHeight; i++){
        //iterate through each of the output columns
        for(int j = 0; j < featureMapWidth; j++){

            float x_l = floor(x_ratio * (float)j);
            float x_h = ceil(x_ratio * (float)j);

            float y_l = floor(x_ratio * (float)i);
            float y_h = ceil(x_ratio * (flat)i);

            float x_weight = (x_ratio * (float)j) - x_l;
            float y_weight = (y_ratio * (float)i) - y_l;

            float a = imageStored[(int)y_l * imageWidth + int(x_l)];
            float b = imageStored[(int)y_l * imageWidth + int(x_h)];
            float c = imageStored[(int)y_h * imageWidth + int(x_l)];
            float d = imageStored[(int)y_h * imageWidth + int(x_h)];

            float pixel = (a + b + c + d)/4;

            featureMapStored[i * featureMapWidth + j].data = pixel;

        }
    }

    //pass interpolated feature map out through stream
    for(int i = 0; i < featureMapWidth * featureMapHeight; i++){
        featureMapStored[i].keep; //indicates whether content of byte of data is processed
        featureMapStored[i].strb; //indicates whether content of byte of data is processed as data or as position
        //featureMapStored[i].user; //not entirely sure what this is used for
        featureMapStored[i].last = 0;
        //featureMapStored[i].id; //identifier (not sure if it is byte or stream)
        //featureMapStored[i].dest; //destination

        //if last byte being sent indicate that in streamed value
        if(i == featureMapWidth * featureMapHeight - 1){
            featureMapStored[i].last = 1;
        }
    }

    featureMap.write(featureMapStored);

}