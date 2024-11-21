#include <hls_stream.h>
#include <ap_int.h>
#include <cmath>
#include <iostream>
#include <random>
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
    ap_uint<32> data
    ap_uint<1> last;    // Indicates last data in a burst
    ap_uint<1> valid;   // Data is valid
    ap_uint<1> ready;   // Consumer is ready to accept data
};

void Interpolation_v2(hls::stream<ap_axiu_1024> &image, hls::stream<ap_axiu_1024> &featureMap);
void bilinear_interpolation(uint8_t (&input_image)[28][28], uint8_t (&output_image)[56][56]);

int main(){

	/*
	 * Initializations
	 */

	//create axi stream for image, featureMap, and loadedInfo
	hls::stream<ap_axiu_1024> image;
	hls::stream<ap_axiu_1024> featureMap;
	//hls::stream<ap_axiu_32> loadedInfo;

	//input image values - tb interpolation
	uint8_t inputImageRed[28][28];
	uint8_t inputImageGreen[28][28];
	uint8_t inputImageBlue[28][28];
	//will need to be broken into 120 bit ints for axi-stream
	uint8_t inputImageAll[28*28*3];

    //bus width: hoping to parameterize this
    const int bitsPerStream = 1024;
    const int pixelsPerStream = bitsPerStream / 8;

	//std::fill_n(inputImageAll, 28*28*3, 1);

	//output image values - tb interpolation
	//uint8_t outputImageRed[56][56];
	//uint8_t outputImageGreen[56][56];
	//uint8_t outputImageBlue[56][56];
	//will need to be reassembled from 120 bit stream outs
	//uint8_t outputImageAll[56*56*3];




	/*
	 * Generate input image
	 */

	int row = 0;
	int column = 0;

	//counting up to 28*28*3 -> 2352
	//generate and store image pixel values
	for(int i = 0; i < 2352; i++){

		//generates a number between 0 and 255
		uint8_t temp = rand() % 256;

		printf("Randomizing value %d, row: %d, column: %d, value: %d \n", i, row, column, temp);

		//used for block being tested
		//inputImageAll[i] = static_cast<uint8_t>(i % 255);
		inputImageAll[i] = temp;


		row = static_cast<int>(floor(i / (28*3)));
		column = i - (row*3);

		/*
		//used for block comparing against one being tested
		if(column % 3 == 0){
			printf(" color red \n");
			inputImageRed[row][static_cast<int>(floor(column/3))] = temp;
		}
		if(column % 3 == 1){
			printf(" color green \n");
			inputImageGreen[row][static_cast<int>(floor(column/3))] = temp;
		}
		if(column % 3 == 2){
			printf(" color blue \n");
			inputImageBlue[row][static_cast<int>(floor(column/3))] = temp;
		}
		*/


	}




	/*
	 * Write input image to Interpolation block
	 */

	//19 writes to give full 28x28 image (with three channels per pixel)
	for(int i = 0; i < 19; i++){

		/*
        //wait until featureMap is ready to receive new signals
        while(!image.ready){
        	sleep(1);
        }
        */

    	image.valid = 0;

    	ap_uint<1024> transValue = 0;

        for(int j = 0; j < pixelsPerStream; j++){ //128 pixel values per transfer

			//if last axi transfer only pass in 48 pixelValues
			if(i == 18 && j >= 48){
				break;
			}

        	//data will be passed in with lowest array value in MSB
        	ap_uint<1024> pixelValue = inputImageAll[i * pixelsPerStream + j];
			transValue = transValue || pixelValue; //add pixel value to transfer
			transValue = transValue << 8; //shift to make room for next pixel value

        }

        printf("Writing to interpolation block, transfer: %d value %d \n", i, valueIn);

        //load data and indicate it is ready to be sent
        image.data = transValue;
        image.valid = 1;
	}


	/*
	 * Read output from Interpolation block axi-stream
	 */


	/*
	//75264 bits in feature map, 588 reads
	for(int i = 0; i < 588; i++){
		printf("Reading value %d from interpolater\n", i);

    	while(image.empty()){
			#pragma HLS PIPELINE II=1
    	}

		ap_uint<128> tempRead;
		uint8_t tempChar;

		tempRead = featureMap.read();

		//16, 8-bit values per transfer
		for(int j = 0; j < 16; j++){

			tempChar = (tempRead & (0xFF << (15 - j))) >> ((15 - j) * 8);
			outputImageAll[i*16 + j] = tempChar;

			row = static_cast<int>(floor(i / (56*3)));
			column = i - (row*3);

			//used for block comparing against one being tested
			if(column % 3 == 0){
				outputImageRed[row][static_cast<int>(floor(column/3))] = tempChar;
			}
			if(column % 3 == 1){
				outputImageGreen[row][static_cast<int>(floor(column/3))] = tempChar;
			}
			if(column % 3 == 2){
				outputImageBlue[row][static_cast<int>(floor(column/3))] = tempChar;
			}
		}
	}
	*/

	/*
	/*
	 * Run bilinear interpolation to test again

	printf("Run bilin interp functions");
	bilinear_interpolation(inputImageRed, outputImageRed);
	bilinear_interpolation(inputImageGreen, outputImageGreen);
	bilinear_interpolation(inputImageBlue, outputImageBlue);

	/*
	 * Verify results


	int errors = 0;
	uint8_t blockValue;
	uint8_t tbValue;

	//OUTPUT IMAGES ARE 2D ARRAYS

	//9408 values to compare
	for(int i = 0; i < 9408; i++){
		printf("Compare value %d \n", i);

		blockValue = outputImageAll[i];


		row = static_cast<int>(floor(i / (56*3)));
		column = i - (row*3);

		//used for block comparing against one being tested
		if(column % 3 == 0){
			tbValue = outputImageRed[row][static_cast<int>(floor(column/3))];
		}
		if(column % 3 == 1){
			tbValue = outputImageGreen[row][static_cast<int>(floor(column/3))];
		}
		if(column % 3 == 2){
			tbValue = outputImageBlue[row][static_cast<int>(floor(column/3))];
		}

		if(blockValue != tbValue){
			errors++;
			printf("Error: value from block = %d, value from tb = %d", blockValue, tbValue);
		}
		else{
			printf("Correct: value from block = %d, value from tb = %d", blockValue, tbValue);
		}

		//gives it chance to interpolate 2 full rows
		if(errors >= 112){
			break;
		}

	}

	*/



}

void bilinear_interpolation(uint8_t (&input_image)[28][28], uint8_t (&output_image)[56][56]){

	printf("Making it to bilin interp function");

    // loop through output image - red
    for (int i = 0; i < 56; ++i) {
        for (int j = 0; j < 56; ++j) {

            // Find the corresponding position in the input image
        	//need to be floats in order to determine neighbors later
            float x = (float)(j) * (28 - 1) / (56 - 1);
            float y = (float)(i) * (28 - 1) / (56 - 1);

            //four nearest pixel values from input
            //consider moving to rounding up on 5 instead of flooring it, success with Diyar's code
            int x1 = std::floor(x);
            int y1 = std::floor(y);
            int x2 = std::min(x1 + 1, 28 - 1);
            int y2 = std::min(y1 + 1, 28 - 1);

            // Bilinear interpolation formula
            int top_left = input_image[y1][x1];
            int top_right = input_image[y1][x2];
            int bottom_left = input_image[y2][x1];
            int bottom_right = input_image[y2][x2];

            output_image[i][j] = (int)((top_left + top_right + bottom_left + bottom_right) /4 );
        }
    }


}
