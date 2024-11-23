#include "Interpolation_v4.h"

// Function to simulate writing data to the stream
void write_data(hls::stream<bus_t>& imageIn, bus_t imageValues[147]) {
    bus_t data;

    for (int i = 0; i < 147; i++) {
        data = imageValues[i];
        imageIn.write(data);
        //std::cout << "Writing data: " << data << std::endl;
    }
}

// Function to simulate reading data from the output stream
void read_data(hls::stream<bus_t>& featureMapOut, bus_t featureMapValues[588]) {
    bus_t data;

    for (int i = 0; i < 588; i++) {

        if (!featureMapOut.empty()) {
            data = featureMapOut.read();
            featureMapValues[i] = data;
            //std::cout << "Feature map output: " << data << std::endl;
        } else {
            std::cout << "Error: No data available in featureMapOut" << std::endl;
        }
    }
}

int main() {
    // Create two streams to simulate the HLS streams
    hls::stream<bus_t> imageIn;
    hls::stream<bus_t> featureMapOut;

    bus_t imageValues[147];
    bus_t featureMapValues[588];

    for(int i = 0; i < 147; i++){

    	bus_t temp = 0;

    	//generates a random 128 bit number by combining 16, 8 bit nums
    	for(int j = 0; j < 16; j++){
			//gets random 8 bit num and adds it to temp value
    		uint8_t randomVal = rand() % 256;
    		//std::cout << "Random Value: " << (int)randomVal << " ";
			temp = temp << 8;
			temp = temp | randomVal;
    	}

    	imageValues[i] = temp;
    	std::cout << "\nCombined 128-bit value: " << temp << std::endl;
    	//std::cout << "\n";
    }

    // Simulate writing data to the imageIn stream
    write_data(imageIn, imageValues);

    // Call the Interpolation_v4 function to read from imageIn and process the data
    Interpolation_v4(imageIn, featureMapOut);

    // Now read and print the data from the featureMapOut stream
    read_data(featureMapOut, featureMapValues);

    for(int i = 0; i < 588; i++){
    	std::cout << "Feature map output: " << featureMapValues[i] << std::endl;
    }

    return 0;
}

