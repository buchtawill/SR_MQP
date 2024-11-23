
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

#define BUS_WIDTH 128

typedef ap_uint<BUS_WIDTH> bus_t;

// Function declarations
void Interpolation_v4(hls::stream<bus_t> &imageIn, hls::stream<bus_t> &featureMapOut);

#endif // INTERPOLATION_H