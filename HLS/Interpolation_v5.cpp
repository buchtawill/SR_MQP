#include "Interpolation_v5.h"

void Interpolation_v5(hls::stream<bus_t> &imageIn, hls::stream<bus_t> &featureMapOut){
	#pragma HLS interface mode=axis port=imageIn
	#pragma HLS interface mode=axis port=featureMapOut


	//Store transfers in 8-bit sections
	uint8_t imageInStoredBytes[imageSize];
	uint8_t imageOutStoredBytes[featureMapSize];

	//Read each transfer in
    for (int i = 0; i < numOfTransfersIn ; i++) {

        // wait until there is data to read
        while (imageIn.empty()) {
            // TO DO: add wait or stall
        }

        bus_t temp = imageIn.read();

        //split read into bytes
        for (int j = 0; j < bytesPerTransfer; j++) {
            uint8_t bitMask = 0xFF;
            imageInStoredBytes[i * bytesPerTransfer + j] = (temp >> ((BUS_WIDTH - 8) - j * 8)) & bitMask;
        }

    }


    for(int i = 0; i < imageSize; i++){
    	for(int j = 0; j < 4; j++){
    		imageOutStoredBytes[i*4 + j] = imageInStoredBytes[i];
    	}
    }


    //Write each transfer out
    for (int i = 0; i < numOfTransfersOut; i++){

    	bus_t temp = 0;

    	//values per bus transfer
    	for(int j = 0; j < bytesPerTransfer; j++){
    		uint8_t currentValue = imageOutStoredBytes[i*bytesPerTransfer + j];
			temp = temp << 8;
			temp = temp | currentValue;
    	}

    	//write stored data to featureMap
        featureMapOut.write(temp);
    }


}
