//throwing errors in VS Code, but I think thats because its not referencing what was downloaded in Vitis HLS
#include <hls_stream.h>
#include <ap_int.h>
#include <cmath>

//creates data type that data will be streamed in through
typedef ap_int<128> stream_data_t;
typedef ap_int<32> lite_data_t;

//all pixel channels at once -> don't output feature map until R, G, and B are all interpolated
//still one stream of pixel data
//stacked in URAM so t

//should be able to store 3 full images (input and output, RGB) in one URAM

//based off of this https://gist.github.com/folkertdev/6b930c7a7856e36dcad0a72a03e66716
//assumes that image and feature map will always be squares, otherwise would need to take in imageHeight
void Interpolation_v1(hls::stream<stream_data_t> &image, hls::stream<stream_data_t> &featureMap, hls::stream<lite_data_t> &loadedInfo){
    #pragma HLS INTERFACE axis port=featureMap
    #pragma HLS INTERFACE axis port=image
    #pragma HLS INTERFACE s_axilite port=loadedInfo
    #pragma HLS INTERFACE axi_cntrl_none port=return //allows Zynq/Microblaze to control IP core

    /*
    const int imageWidth = 28;
    const int imageHeight = 28;
    const int featureMapWidth = 56;
    const int featureMapHeight = 56;
    */

    //we should be able to parameterize this, but starting with constant for now because I don't want to mess up data type
    const ap_uint<8> pixelsPerStream = 128;

    //store value from axi stream
    ap_uint<8> loadValue = loadedInfo.read();

    //mask and store values for image width and scaling factor when image width is the LSByte and scaling factor is next LSByte
    ap_uint<8> imageWidth = loadValue & 0377;
    ap_uint<8> scalingFactor = loadValue & 0177400;

    //calculate size of image and feature map
    ap_uint<8> imageSize = imageWidth*imageWidth;
    ap_uint<8> featureMapWidth = imageWidth*scalingFactor;

    //need to store image value streamed in so that it can be used for bilinear interpolation
    //ap_uint<8> imageStored[imageWidth*imageWidth];
    ap_uint<8> imageStored[32*32];
    //ap_uint<8> featureMapStored[imageWidth*scalingFactor*imageWidth*scalingFactor];
    ap_uint<8> featureMapStored[28*2*28*2];

    //store input image and feature maps in URAM
    #pragma HLS RESOURCE variable=imageStored core=XPM_MEMORY uram
    #pragma HLS RESOURCE variable=featureMapStored code=XPM_MEMORY uram

    //arrays to temporarily store image values for processing, need 2 rows to complete interpolation
    //then toss out the first, move up the second, and replace the second
    //ap_uint<8> tempRed[imageWidth*2];
    //ap_uint<8> tempGreen[imageWidth*2];
    //ap_uint<8> tempBlue[imageWidth*2];
    ap_uint<8> tempRed[28*2];
    ap_uint<8> tempGreen[28*2];
    ap_uint<8> tempBlue[28*2];

    while(true){

        //128 bits, 16 pixel values, 5 images and an extra
        //might be easier to only use 120
        //read in and store the image values in staging ground
        for(int i = 0; i < pixelsPerStream / (8*3); i++){
            tempRed[i] = (image.read() >> i*8) & 0377;
            tempGreen[i] = (image.read() >> i*8*2) & 0377;
            tempBlue[i] = (image.read() >> i*8*3) & 0377;
        }

        //staging areas for each RGB
        //read URAM to populate staging areas
        //perform operations
        //load back into URAM and pull more out
        //store both input and output image in URAM


        //integer intervals: determines the location of the interpolated points
        float ratio = (imageWidth - 1) / (featureMapWidth - 1);

        //iterate through each of the output rows
        for(int i = 0; i < featureMapWidth; i++){
            //iterate through each of the output columns
            for(int j = 0; j < featureMapWidth; j++){

                //determies x values for the coordinates to the left and right of the pixel being interpolated
                //uint_8
            	ap_uint<8> x_l = floor(ratio * (float)j);
            	ap_uint<8> x_h = ceil(ratio * (float)j);

                //determines the y values for the coordinate to the top and bottom of the pixel being interpolated
            	ap_uint<8> y_l = floor(ratio * (float)i);
            	ap_uint<8> y_h = ceil(ratio * (float)i);


                //gets the values from the four pixels you are interpolating from
            	ap_uint<8> a = imageStored[y_l * imageWidth + x_l];
            	ap_uint<8> b = imageStored[y_l * imageWidth + x_h];
            	ap_uint<8> c = imageStored[y_h * imageWidth + x_l];
            	ap_uint<8> d = imageStored[y_h * imageWidth + x_h];

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
