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
    const ap_uint<8> bitsPerStream = 128;

    //store value from axi stream
    ap_uint<8> loadValues[32] = loadedInfo.read();

    //mask and store values for image width and scaling factor when image width is the LSByte and scaling factor is next LSByte
    //0 in front indicates that its in hex
    ap_uint<8> imageWidth = loadValue & 0xFF;
    ap_uint<8> scalingFactor = loadValue & 0xFF00;

    //calculate size of image and feature map
    ap_uint<8> imageSize = imageWidth*imageWidth;
    ap_uint<8> featureMapWidth = imageWidth*scalingFactor;

    //need to store image value streamed in so that it can be used for bilinear interpolation
    //ap_uint<8> imageStored[imageWidth*imageWidth];
    ap_uint<8> imageStored[32*32];
    //ap_uint<8> featureMapStored[imageWidth*scalingFactor*imageWidth*scalingFactor];
    //storing the feature maps separately allows for more pipelining
    ap_uint<8> featureMapStoredRed[28*2*28*2];
    ap_uint<8> featureMapStoredGreen[28*2*28*2];
    ap_uint<8> featureMapStoredBlue[28*2*28*2];

    //integer intervals: determines the location of the interpolated points
    float ratio = (imageWidth - 1) / (featureMapWidth - 1);

    //store input image and feature maps in URAM
    #pragma HLS RESOURCE variable=imageStored core=XPM_MEMORY uram
    #pragma HLS RESOURCE variable=featureMapStoredRed code=XPM_MEMORY uram
    #pragma HLS RESOURCE variable=featureMapStoredGreen code=XPM_MEMORY uram
    #pragma HLS RESOURCE variable=featureMapStoredBlue code=XPM_MEMORY uram

    //arrays to temporarily store image values for processing, need 2 rows to complete interpolation
    //then toss out the first, move up the second, and replace the second
    unsigned char tempRed[28*2];
    unsigned char tempGreen[28*2];
    unsigned char tempBlue[28*2];
    //number of pixels stored, 2^10 = 1024, first power of 2 greater than 28*28*3
    ap_uint<12> valueStored = 0;
    ap_uint<12> fullRowsStored = 0;
    up_uint<12> fullRowsUpscaled = 0;

    while(true){

        unsigned char imageLoadIn[16] = image.read();
        
        //could go up to i < 16, but not sure how to handle storing one extra color value
        for(int i = i; i < 15; i++){

            if(i % 3 == 0){
                tempRed[valueStored + i] = imageLoadIn[floor(i/3)];
            }
            else if(i % 3 == 1){
                tempGreen[valuedStored + i] = imageLoadIn[floor(i/3)];
            }
            else if(i % 3 == 2){
                tempBlue[valueStored + i] == imageLoadIn[floor(i/3)];
            }
        }

        //it seems like this will work in the way C++ is compiled/exectuted but might break it
        valueStored += 5;
        fullRowsStored = fullRowsStored + floor(valuedStore / 28)

        //once two rows have been stored
        if(fullRowsStored == 2){

            //makes the loops execute concurrently
            #pragma HLS DATAFLOW

            //iterate through the two stored rows
            for(int i = 0; i < 2; i++){
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
                    ap_uint<8> a = tempRed[y_l * imageWidth + x_l];
                    ap_uint<8> b = tempRed[y_l * imageWidth + x_h];
                    ap_uint<8> c = tempRed[y_h * imageWidth + x_l];
                    ap_uint<8> d = tempRed[y_h * imageWidth + x_h];

                    //don't need to multiply by weights since they are all equadistant
                    int pixel = int((a + b + c + d)/4);

                    featureMapStoredRed[i * featureMapWidth + j] = pixel;
                }
            }

            //iterate through the two stored rows
            for(int i = 0; i < 2; i++){
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
                    ap_uint<8> a = tempGreen[y_l * imageWidth + x_l];
                    ap_uint<8> b = tempGreen[y_l * imageWidth + x_h];
                    ap_uint<8> c = tempGreen[y_h * imageWidth + x_l];
                    ap_uint<8> d = tempGreen[y_h * imageWidth + x_h];

                    //don't need to multiply by weights since they are all equadistant
                    int pixel = int((a + b + c + d)/4);

                    featureMapStoredGreen[i * featureMapWidth + j] = pixel;
                }
            }

                        //iterate through the two stored rows
            for(int i = 0; i < 2; i++){
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
                    ap_uint<8> a = tempBlue[y_l * imageWidth + x_l];
                    ap_uint<8> b = tempBlue[y_l * imageWidth + x_h];
                    ap_uint<8> c = tempBlue[y_h * imageWidth + x_l];
                    ap_uint<8> d = tempBlue[y_h * imageWidth + x_h];

                    //don't need to multiply by weights since they are all equadistant
                    int pixel = int((a + b + c + d)/4);

                    featureMapStoredBlue[i * featureMapWidth + j] = pixel;
                }
            }

            fullRowsUpscaled += 1;
        }


        if(fullRowsUpscaled == featureMapWidth){

            for(int i = 0; i < featureMapWidth; i++){

                for(int j = 0; j < featureMapWidth; j++){
                    if(j % 3 == 0){
                        featureMap.write(tempRed[i*featureMapWidth + j]);
                    }
                    else if(i % 3 == 1){
                        featureMap.write(tempGreen[i*featureMapWidth + j]);
                    }
                    else if(i % 3 == 2){
                        featureMap.write(tempBlue[i*featureMapWidth + j]);
                    }
                }
            }
        }

    } //while true

}
