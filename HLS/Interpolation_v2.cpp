#include <hls_stream.h>
#include <ap_int.h>
#include <cmath>
#include <iostream>
#include <cstdint>
#include <chrono>
#include <thread>

// AXI-stream data type (1024-bit)
struct ap_axiu_1024 {
    ap_uint<1024> data; //sender -> receiver
    ap_uint<1> last;    //sender -> receiver: Indicates last data in a burst
    ap_uint<1> valid;   //sender -> receiver: Data is valid
    ap_uint<1> ready;   //receiver -> sender: receiver is ready to accept data
};

// AXI-lite data type (32-bit)
struct ap_axiu_32 {
    ap_uint<32> data;
    ap_uint<1> last;    // Indicates last data in a burst
    ap_uint<1> valid;   // Data is valid
    ap_uint<1> ready;   // Consumer is ready to accept data
};

//void Interpolation_v2(hls::stream<stream_data_t> &image, hls::stream<stream_data_t> &featureMap, hls::stream<lite_data_t> &loadedInfo){
void Interpolation_v2(hls::stream<ap_axiu_1024> &image, hls::stream<ap_axiu_1024> &featureMap){
    #pragma HLS INTERFACE axis port=featureMap
    #pragma HLS INTERFACE axis port=image
    //#pragma HLS INTERFACE s_axilite port=loadedInfo
    //#pragma HLS INTERFACE axi_cntrl_none port=return //allows Zynq/Microblaze to control IP core

    //store value from axi lite
    //ap_axiu_32 loadValues;
    //loadValues = loadedInfo.read();

    //mask and store values for image width and scaling factor when image width is the LSByte and scaling factor is next LSByte
    //0 in front indicates that its in hex
    //ap_uint<8> imageWidth = loadValues & 0xFF;
    //ap_uint<8> scalingFactor = loadValues & 0xFF00;
    
    //bus width: hoping to parameterize this
    const int bitsPerStream = 1024;
    const int pixelsPerStream = bitsPerStream / 8;

    //input image widths: hoping to parameterize this
    const int imageWidth = 28;
    const int imageHeight = 28;
    const int channels = 3;
    const int upscalingFactor = 2;

    //dimensions of feature<Map
    int featureMapWidth = imageWidth * upscalingFactor;
    int featureMapHeight = imageHeight * upscalingFactor;

    //need to store image value streamed in so that it can be used for bilinear interpolation
    uint8_t imageStored[imageWidth*imageHeight*3];
    //std::fill_n(imageStored, imageWidth*imageHeight*3, 1);
    
    /*
    for(int i = 0; i < imageWidth * imageHeight * 3; i++){
    	printf("%d", imageStored[i]);
    	if(i % 10 == 0){
    		printf("\n");
    	}
    }
    */


    //need to store featureMap value to be streamed out
    uint8_t featureMapStored[featureMapWidth * featureMapHeight];
    //uint8_t featureMapStored[56 * 56*3];

    //consider storing in block ram instead of ultraram
    //#pragma HLS bind_storage variable=imageStored type=RAM_2P impl = URAM
    //#pragma HLS bind_storage variable=featureMapStored type=RAM_2P impl = URAM

    ap_axiu_1024 imageLoadIn; //value from axi-stream
    ap_uint<1024> dataLoadIn; //data value from axi-stream

	//#pragma HLS ARRAY_PARTITION variable=imageStored complete
	//#pragma HLS ARRAY_PARTITION variable=featureMapStored complete


    while(true){

        //19 reads to get full 28x28 image (with three channels per pixel)
        for(int i = 0; i < 19; i++){

			//loads in image from axi-stream to array of uint_8
			imageLoadIn = image.read();
			printf("Got value %d \n", imageLoadIn);

			/*
            //wait until image has new data
            while(!imageLoadIn.valid && !imageLoadIn.last){
            	std::this_thread::sleep_for(std::chrono::nanoseconds(1));
            }
            */

            if(imageLoadIn.valid && imageLoadIn.last)

        	printf("reading value %d from test bench \n", i);

            imageLoadIn.ready = 0;


			dataLoadIn = imageLoadIn.data;

			//break data into 8 bit chunks
			for(int j = 0; j < pixelsPerStream; j++){

				//last axi transfer will only pass in 48 pixelValues
				if(i == 18 && j >= 48){
					break;
				}

				//data will be passed in with lowest array value in MSB
				ap_uint<1024> bitMask = 0xFF << 1016;
				uint8_t bitValue = (dataLoadIn && bitMask) >> 1016;
				imageStored[i*pixelsPerStream + j] = bitValue;
				dataLoadIn = dataLoadIn << 8;

			}

			//tell sender that module is ready for next transfer
			imageLoadIn.ready = 1;
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


        //74 transfers of 128 bits to pass out whole feature map
        for(int i = 0; i < 74; i++){

        	ap_axiu_1024 featureMapLoadOut = featureMap.read();

            //wait until featureMap is ready to receive new signals
        	/*
            while(!featureMapLoadOut.ready){
            	std::this_thread::sleep_for(std::chrono::nanoseconds(1));
            }
            */

            featureMapLoadOut.valid = 0;

        	ap_uint<1024> transValue = 0;

            for(int j = 0; j < pixelsPerStream; j++){ //128 pixel values per transfer

				//if last axi transfer only pass in 64 pixelValues
				if(i == 73 && j >= 64){
					break;
				}

            	//data will be passed in with lowest array value in MSB
            	ap_uint<1024> pixelValue = featureMapStored[i * pixelsPerStream + j];
				transValue = transValue || pixelValue; //add pixel value to transfer
				transValue = transValue << 8; //shift to make room for next pixel value


            }

            //load data and indicate it is ready to be sent
            featureMapLoadOut.data = transValue;
            featureMapLoadOut.valid = 1;

        }

    } //end while true
}
