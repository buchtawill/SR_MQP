#include "Interpolation_v5.h"

// Function to simulate writing data to the stream
void write_data(hls::stream<bus_t>& imageIn, bus_t imageValues[numOfTransfersIn]) {
    bus_t data;

    for (int i = 0; i < numOfTransfersIn; i++) {
        data = imageValues[i];
        imageIn.write(data);
        //std::cout << "Writing data: " << data << std::endl;
    }
}

// Function to simulate reading data from the output stream
void read_data(hls::stream<bus_t>& featureMapOut, bus_t featureMapValues[numOfTransfersOut]) {
    bus_t data;

    for (int i = 0; i < numOfTransfersOut; i++) {

        if (!featureMapOut.empty()) {
            data = featureMapOut.read();
            featureMapValues[i] = data;
            //std::cout << "Feature map output: " << data << std::endl;
        } else {
            std::cout << "Error: No data available in featureMapOut" << std::endl;
        }
    }
}



int main(){

	//Create two axi-streams to get values to and from module
    hls::stream<bus_t> imageIn;
    hls::stream<bus_t> featureMapOut;

    //Store 32-bit values passed in to and out of module
    bus_t imageValuesTransfer[numOfTransfersIn];
    bus_t featureMapValuesTransfer[numOfTransfersOut];

    //Hardcoded interpolation values for test
	uint8_t input_test1[] =  {
		255, 0, 0,
		0, 255, 0,
		0, 0, 255,
		255, 255, 255
		};

	uint8_t output_test1[] = {
		// Row 0
		255, 0, 0,        // (0,0)
		128, 128, 0,      // (1,0)
		0, 255, 0,        // (2,0)
		0, 255, 0,        // (3,0)

		// Row 1
		128, 0, 128,      // (0,1)
		128, 128, 128,    // (1,1)
		128, 255, 128,    // (2,1)
		128, 255, 128,    // (3,1)

		// Row 2
		0, 0, 255,        // (0,2)
		128, 128, 255,    // (1,2)
		255, 255, 255,    // (2,2)
		255, 255, 255,    // (3,2)

		// Row 3
		0, 0, 255,        // (0,3)
		128, 128, 255,    // (1,3)
		255, 255, 255,    // (2,3)
		255, 255, 255     // (3,3)
	};

    //Assemble 32 bit image transfer values
    for(int i = 0; i < numOfTransfersIn; i++){

    	bus_t temp = 0;

    	//create 32 bit bus value by combining 4, 8 bit values
    	for(int j = 0; j < bytesPerTransfer; j++){
    		uint8_t currentValue = input_test1[i*bytesPerTransfer + j];
			temp = temp << 8;
			temp = temp | currentValue;
    	}

    	imageValuesTransfer[i] = temp;
    }
    //printf("making it through assembly\n");

    // Simulate writing data to the imageIn stream
    write_data(imageIn, imageValuesTransfer);
    //printf("making it through write \n");

    // Call the Interpolation_v4 function to read from imageIn and process the data
    Interpolation_v5(imageIn, featureMapOut);
    //printf("making it through call \n");

    // Now read and print the data from the featureMapOut stream
    read_data(featureMapOut, featureMapValuesTransfer);
    //printf("making it through read \n");

    uint8_t bitMask = 0xFF;
    uint8_t featureMapValues[imageSize];

    //disassemble feature map transfer values
    for(int i = 0; i < numOfTransfersOut; i++){
    	//split each 32 bit bus into its 8 bit pieces
        for (int j = 0; j < bytesPerTransfer; j++) {
            featureMapValues[i*bytesPerTransfer + j] = (featureMapValuesTransfer[i] >> ((BUS_WIDTH - 8) - j * 8)) & bitMask;
        }
    }
    //printf("making it through disassemble of feature map \n");

    int errors = 0;

    for(int i = 0; i < imageSize; i++){

    	for(int j = 0; j < 4; j++){
			printf("Expected Value: %d, Received Value: %d\n", input_test1[i], featureMapValues[i*4 + j]);
			if(input_test1[i] != featureMapValues[i*4 + j]){
				errors++;
			}
    	}
    }

    if(errors){
    	printf("ERROR!\n");
    }
    else{
    	printf("SUCCESS!\n");
    }

}
