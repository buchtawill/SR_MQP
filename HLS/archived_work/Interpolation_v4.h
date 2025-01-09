
#ifndef INTERPOLATION_H // Include guard
#define INTERPOLATION_H 

// Libraries
#include "ap_int.h"
#include "hls_stream.h"
#include <stdio.h>
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm> 

//#define BUS_WIDTH 128
#define BUS_WIDTH 16
#define IMAGE_WIDTH 2
#define IMAGE_HEIGHT 2
#define NUM_OF_CHANNELS 1
#define SCALING_FACTOR 2

const int imageSize = IMAGE_WIDTH * IMAGE_HEIGHT * NUM_OF_CHANNELS;
const int imageBitSize = imageSize * 8;
const int numOfTransfersIn = imageBitSize / BUS_WIDTH;
const int featureMapWidth = IMAGE_WIDTH * SCALING_FACTOR;
const int featureMapHeight = IMAGE_HEIGHT * SCALING_FACTOR;
const int featureMapSize = featureMapWidth * featureMapHeight * NUM_OF_CHANNELS;
const int featureMapBitSize = featureMapSize * 8;
const int numOfTransfersOut = featureMapBitSize / BUS_WIDTH;
const int bytesPerTransfer = BUS_WIDTH / 8;

typedef ap_uint<BUS_WIDTH> bus_t;

// Function declarations
void Interpolation_v4(hls::stream<bus_t> &imageIn, hls::stream<bus_t> &featureMapOut);

#endif // INTERPOLATION_H