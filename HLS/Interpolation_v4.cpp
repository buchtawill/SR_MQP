#include "Interpolation_v4.h"

void Interpolation_v4(hls::stream<bus_t> &imageIn, hls::stream<bus_t> &featureMapOut){
	#pragma HLS interface mode=axis port=imageIn
	#pragma HLS interface mode=axis port=featureMapOut

	const int inputWidth = 28;
	const int numOfChannels = 3;
	const int numOfTransfers = inputWidth * inputWidth * numOfChannels * 8 / BUS_WIDTH;

	//store 128-bit transfers
	bus_t allImageInStored[147];
	bus_t allImageOutStored[588];

	//store in 8-bit sections
	uint8_t imageInStoredBytes[2352];
	uint8_t imageOutStoredBytes[9408];

    // 147 transfers are needed to get the full image
    for (int i = 0; i < 147; i++) {

        // wait until there is data to read
        while (imageIn.empty()) {
            // TO DO: add wait or stall
        }

        // read from stream
        //allImageInStored[i] = imageIn.read();

        bus_t temp = imageIn.read();

        for (int j = 0; j < 16; j++) {
            uint8_t bitMask = 0xFF;
            imageInStoredBytes[i * 16 + j] = (temp >> (120 - j * 8)) & bitMask;
        }

        /*
        //16 bytes per transfer
        for(int j = 0; j < 16; j++){
        	uint8_t bitMask = 0xFF;
        	bus_t shifted = temp >> (120 - j*8);
        	shifted = shifted & bitMask;
        	imageInStoredBytes[i*16 + j] = shifted;
        }
        */
    }

    //INTERPOLATION BLOCK WILL GO HERE
    for(int i = 0; i < 9408; i++){
    	imageOutStoredBytes[i] = imageInStoredBytes[static_cast<int>(floor(i / 4))];
    }


    for (int i = 0; i < 588; i++){

    	bus_t temp = 0;

    	//combines 8 bytes from input image for featureMap output
    	for(int j = 0; j < 16; j++){
			//gets random 8 bit num and adds it to temp value
    		uint8_t pixelValue = imageOutStoredBytes[i*16 + j];
    		//std::cout << "Random Value: " << (int)randomVal << " ";
			temp = temp << 8;
			temp = temp | pixelValue;
    	}

    	//write stored data to featureMap
        featureMapOut.write(temp);
    }

}
