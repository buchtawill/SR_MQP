// #include "VInterpolation_v1.h"     // Include Verilated model header
// #include "verilated.h"             // Include Verilator utilities
// #include "verilated_vcd_c.h"       // Include Verilator trace file format header
#include <iostream>                // For std::cout
#include <iomanip>                 // For std::setw
#include <vector>                  // For std::vector
#include <cmath>                   // For bilinear interpolation
#include <algorithm>              // For std::min
#include <cstdint>                 // For uint8_t
#include <random>

/*
verilator --cc Interpolation_v1.v --exe tb_interpolation.cpp --timing -Wall --Wno-WIDTH --Wno-UNOPTFLAT --Wno-UNSIGNED --Wno-IMPLICIT --Wno-LITENDIAN --Wno-DECLFILENAME --Wno-GENUNNAMED --Wno-TIMESCALEMOD --Wno-PINCONNECTEMPTY
*/

struct TestCase {
    std::vector<uint8_t> input_image;
    int width;
    int height;
    int new_width;
    int new_height;
    int channels;
    std::vector<uint8_t> expected_output;
};

// Since uint128_t is not standard, use two uint64_t for TDATA
typedef struct {
    uint64_t low;
    uint64_t high;
} TDATA_t;

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
uint8_t input_test2[] = {
    // Row 0
    10, 20, 30,    // (0,0)
    40, 60, 80,    // (1,0)
    // Row 1
    70, 90, 110,   // (0,1)
    130, 150, 170  // (1,1)
};

uint8_t output_test2[] = {
    // Row 0
    10, 20, 30,      // (0,0)
    25, 40, 55,      // (1,0)
    40, 60, 80,      // (2,0)
    40, 60, 80,      // (3,0)

    // Row 1
    40, 55, 70,      // (0,1)
    63, 80, 98,      // (1,1)
    85, 105, 125,    // (2,1)
    85, 105, 125,    // (3,1)

    // Row 2
    70, 90, 110,     // (0,2)
    100, 120, 140,   // (1,2)
    130, 150, 170,   // (2,2)
    130, 150, 170,   // (3,2)

    // Row 3
    70, 90, 110,     // (0,3)
    100, 120, 140,   // (1,3)
    130, 150, 170,   // (2,3)
    130, 150, 170    // (3,3)
};

uint8_t input_test3[] = {
    // Row 0
    200, 50, 25,     // (0,0)
    100, 150, 75,    // (1,0)

    // Row 1
    50, 200, 125,    // (0,1)
    25, 100, 175     // (1,1)
};

uint8_t output_test3[] = {
    // Row 0
    200, 50, 25,     // (0,0)
    150, 100, 50,    // (1,0)
    100, 150, 75,    // (2,0)
    100, 150, 75,    // (3,0)

    // Row 1
    125, 125, 75,    // (0,1)
    94, 125, 100,    // (1,1)
    63, 125, 125,    // (2,1)
    63, 125, 125,    // (3,1)

    // Row 2
    50, 200, 125,    // (0,2)
    38, 150, 150,    // (1,2)
    25, 100, 175,    // (2,2)
    25, 100, 175,    // (3,2)

    // Row 3
    50, 200, 125,    // (0,3)
    38, 150, 150,    // (1,3)
    25, 100, 175,    // (2,3)
    25, 100, 175     // (3,3)
};

uint8_t input_test4[] = {
    // Row 0
    15, 45, 75,    // (0,0)
    85, 115, 145,  // (1,0)

    // Row 1
    155, 185, 215, // (0,1)
    225, 255, 35   // (1,1)
};

uint8_t output_test4[] = {
    // Row 0
    15, 45, 75,      // (0,0)
    50, 80, 110,     // (1,0)
    85, 115, 145,    // (2,0)
    85, 115, 145,    // (3,0)

    // Row 1
    85, 115, 145,    // (0,1)
    120, 150, 118,   // (1,1)
    155, 185, 90,    // (2,1)
    155, 185, 90,    // (3,1)

    // Row 2
    155, 185, 215,   // (0,2)
    190, 220, 125,   // (1,2)
    225, 255, 35,    // (2,2)
    225, 255, 35,    // (3,2)

    // Row 3
    155, 185, 215,   // (0,3)
    190, 220, 125,   // (1,3)
    225, 255, 35,    // (2,3)
    225, 255, 35     // (3,3)
};


uint8_t input_test5[] = {
    // Row 0
    10, 20, 30,    // (0,0) - Top-left pixel
    40, 50, 60,    // (1,0) - Top-right pixel

    // Row 1
    70, 80, 90,    // (0,1) - Bottom-left pixel
    100, 110, 120  // (1,1) - Bottom-right pixel
};

uint8_t output_test5[] = {
    // Row 0
    10, 20, 30,    // (0,0)
    25, 35, 45,    // (1,0)
    40, 50, 60,    // (2,0)
    40, 50, 60,    // (3,0)

    // Row 1
    40, 50, 60,    // (0,1)
    55, 65, 75,    // (1,1)
    70, 80, 90,    // (2,1)
    70, 80, 90,    // (3,1)

    // Row 2
    70, 80, 90,    // (0,2)
    85, 95, 105,   // (1,2)
    100, 110, 120, // (2,2)
    100, 110, 120, // (3,2)

    // Row 3
    70, 80, 90,    // (0,3)
    85, 95, 105,   // (1,3)
    100, 110, 120, // (2,3)
    100, 110, 120  // (3,3)
};

uint8_t input_test6[] = {
    // Row 0
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 175, 191, 191, 176, 190, 197, 175, 190, 196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 176, 191, 197, 176, 190, 197, 175, 191, 191, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 175, 191, 196, 176, 190, 197, 175, 190, 196, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 175, 190, 196, 176, 190, 197, 175, 191, 196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 65, 160, 71, 67, 159, 71, 67, 159, 70, 67, 159, 70, 115, 172, 125, 176, 190, 197, 
 167, 186, 186, 67, 159, 70, 67, 159, 70, 67, 159, 70, 67, 159, 70, 67, 159, 70, 67, 159, 70, 67, 159, 70, 67, 159, 70, 167, 186, 186, 
 176, 190, 197, 115, 172, 125, 67, 159, 70, 67, 159, 70, 67, 160, 71, 66, 157, 72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 66, 159, 71, 67, 160, 71, 67, 160, 71, 67, 160, 71, 105, 159, 116, 176, 190, 197, 166, 184, 185, 63, 153, 67, 67, 160, 71, 67, 160, 
 71, 67, 160, 71, 67, 160, 71, 67, 160, 71, 67, 160, 71, 62, 152, 67, 166, 184, 185, 176, 190, 197, 105, 160, 116, 67, 160, 71, 67, 160, 
 71, 67, 160, 71, 67, 160, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 165, 75, 67, 160, 71, 67, 160, 71, 67, 160, 71, 67, 160, 
 71, 96, 151, 107, 176, 190, 197, 161, 182, 179, 58, 144, 61, 67, 160, 71, 67, 160, 71, 67, 160, 71, 67, 160, 71, 67, 160, 71, 67, 160, 
 71, 58, 144, 62, 161, 182, 179, 176, 190, 197, 95, 151, 106, 67, 160, 71, 67, 160, 71, 67, 160, 71, 67, 160, 71, 63, 159, 63, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 157, 72, 67, 160, 71, 67, 160, 71, 67, 160, 71, 67, 160, 71, 61, 142, 66, 111, 158, 123, 82, 144, 90, 64, 
 155, 68, 67, 160, 71, 67, 160, 71, 67, 160, 71, 67, 160, 71, 67, 160, 71, 67, 160, 71, 64, 156, 69, 81, 144, 90, 110, 157, 122, 61, 
 142, 66, 67, 160, 71, 67, 160, 71, 67, 160, 71, 67, 160, 71, 72, 157, 72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 103, 170, 103, 67, 160, 
 71, 67, 160, 71, 67, 160, 71, 67, 160, 71, 67, 160, 71, 65, 157, 69, 66, 159, 71, 67, 160, 71, 67, 160, 71, 67, 160, 71, 67, 160, 71, 
 67, 160, 71, 67, 160, 71, 67, 160, 71, 67, 160, 71, 66, 159, 71, 65, 156, 69, 67, 160, 71, 67, 160, 71, 67, 160, 71, 67, 160, 71, 67, 
 160, 71, 103, 170, 103, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 130, 202, 143, 76, 164, 79, 76, 164, 79, 76, 164, 79, 76, 164, 79, 76, 164,
  79, 76, 164, 79, 76, 164, 79, 76, 164, 79, 76, 164, 79, 76, 164, 79, 76, 164, 79, 76, 164, 79, 76, 164, 79, 76, 164, 79, 76, 164, 79, 
  76, 164, 79, 76, 164, 79, 76, 164, 79, 76, 164, 79, 76, 164, 79, 76, 164, 79, 76, 164, 79, 130, 202, 143, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 194, 230, 206, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200,
   230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201,
    200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 194, 230, 206, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 194, 230, 206, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 
    201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 
    230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 194, 230, 206, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 194, 230, 206, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 
    230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 
    200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 194, 230, 206, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 194, 230, 206, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 195, 227, 197, 
    142, 191, 145, 121, 177, 124, 144, 192, 146, 196, 228, 197, 200, 230, 201, 200, 230, 201, 198, 229, 199, 151, 196, 152, 122, 177, 
    124, 138, 188, 141, 193, 225, 194, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 194, 230, 206, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 194, 230, 206, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 197, 227, 198, 81, 148, 84, 
    46, 125, 50, 57, 132, 60, 46, 125, 50, 90, 155, 94, 200, 230, 201, 200, 230, 201, 111, 170, 113, 46, 125, 50, 51, 129, 55, 46, 125, 
    50, 74, 143, 77, 197, 227, 198, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 194, 230, 206, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 194, 230, 206, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 155, 199, 157, 46, 125, 50, 105, 165, 108, 
    198, 229, 199, 66, 139, 70, 47, 126, 51, 193, 225, 194, 200, 230, 201, 58, 134, 63, 54, 131, 57, 195, 227, 197, 89, 155, 93, 46, 
    125, 50, 172, 212, 174, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 194, 230, 206, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 194, 230, 206, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 192, 224, 194, 179, 216, 181, 191, 223, 192, 195,
    226, 196, 58, 134, 63, 57, 132, 60, 198, 229, 199, 200, 230, 201, 87, 152, 90, 48, 126, 52, 145, 193, 148, 68, 140, 72, 56, 132, 
    60, 190, 223, 191, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 194, 230, 206, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    194, 230, 206, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 106, 165, 
    109, 46, 125, 50, 137, 186, 139, 200, 230, 201, 200, 230, 201, 162, 204, 163, 48, 126, 52, 46, 125, 50, 46, 125, 50, 123, 177, 
    126, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 194, 230, 206, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    194, 230, 206, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 120, 175, 122, 46, 125, 50, 
    114, 171, 116, 200, 230, 201, 200, 230, 201, 196, 227, 196, 59, 134, 63, 64, 137, 67, 168, 207, 169, 91, 156, 93, 47, 126, 51, 171, 210, 172, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 194, 230, 206, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 194, 230, 206, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 131, 183, 134, 46, 125, 50, 96, 159, 99, 198, 229, 199, 200, 230, 201, 200, 230, 201, 187, 221, 188, 46, 125, 50, 78, 147, 81, 199, 229, 200, 116, 173, 118, 46, 125, 50, 148, 195, 151, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 194, 230, 206, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 194, 230, 206, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 171, 210, 172, 46, 125, 50, 46, 125, 50, 54, 131, 57, 56, 132, 60, 56, 132, 60, 157, 201, 159, 200, 230, 201, 84, 152, 87, 46, 125, 50, 64, 138, 68, 47, 127, 51, 58, 134, 63, 187, 221, 188, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 194, 230, 206, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 194, 230, 206, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 184, 219, 185, 128, 181, 130, 128, 181, 130, 128, 181, 130, 128, 181, 130, 128, 181, 130, 179, 215, 180, 200, 230, 201, 195, 226, 196, 138, 188, 141, 112, 170, 115, 127, 180, 130, 184, 219, 185, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 194, 230, 206, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 201, 228, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 201, 228, 201, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 199, 229, 200, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 198, 228, 202, 199, 229, 200, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 200, 230, 201, 201, 228, 201, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 195, 225, 195, 195, 225, 195, 195, 225, 195, 195, 225, 195, 195, 225, 195, 195, 225, 195, 195, 225, 195, 195, 225, 195, 195, 225, 195, 195, 225, 195, 195, 225, 195, 195, 225, 195, 195, 225, 195, 195, 225, 195, 195, 225, 195, 195, 225, 195, 195, 225, 195, 195, 225, 195, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0
};

uint8_t output_test6[] = {

};

// // Simulation time variables
// vluint64_t main_time = 0;    // Current simulation time
// const vluint64_t sim_time = 1000000;  // Adjust as needed

// double sc_time_stamp() {     // Called by $time in Verilog
//     return main_time;
// }

// Clock period in time units
#define CLK_PERIOD 10

// Global module pointer
// VInterpolation_v1* top;

// Function to toggle the clock and evaluate the module
void tick() {
    // top->ap_clk = 0;
    // top->eval();
    // main_time += CLK_PERIOD / 2;
    // top->ap_clk = 1;
    // top->eval();
    // main_time += CLK_PERIOD / 2;
}

// // Initialize the module
// void initialize() {
//     top->ap_rst_n = 0;  // Active-low reset
//     top->ap_start = 0;
//     top->image_r_TVALID = 0;
//     top->featureMap_TREADY = 0;
//     for (int i = 0; i < 5; ++i) {
//         tick();
//     }
//     top->ap_rst_n = 1;  // Deassert reset
//     for (int i = 0; i < 5; ++i) {
//         tick();
//     }
// }

// void start_module() {
//     top->ap_start = 1;
//     tick();
//     top->ap_start = 0;  // Pulse ap_start
// }

// void wait_for_module_done() {
//     while (!top->ap_done && main_time < sim_time) {
//         tick();
//     }
// }

TDATA_t pack_tdata(const uint8_t* bytes) {
    TDATA_t tdata;
    tdata.low = 0;
    tdata.high = 0;
    for (int i = 0; i < 8; ++i) {
        tdata.low |= ((uint64_t)bytes[i]) << (8 * i);
        tdata.high |= ((uint64_t)bytes[i + 8]) << (8 * i);
    }
    return tdata;
}

void unpack_tdata(const TDATA_t& tdata, uint8_t* bytes) {
    for (int i = 0; i < 8; ++i) {
        bytes[i] = (tdata.low >> (8 * i)) & 0xFF;
        bytes[i + 8] = (tdata.high >> (8 * i)) & 0xFF;
    }
}

// Function to generate a random image
std::vector<uint8_t> generate_random_image(int width, int height, int channels) {
    std::vector<uint8_t> image(width * height * channels);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    for (auto& pixel : image) {
        pixel = dis(gen);
    }

    return image;
}

std::vector<uint8_t> bilinearInterpolation(
    const std::vector<uint8_t>& image,
    int width,
    int height,
    int channels,
    float scale)
{
    // Calculate new dimensions
    int newWidth = static_cast<int>(width * scale);
    int newHeight = static_cast<int>(height * scale);

    // Initialize the output image
    std::vector<uint8_t> outputImage(newWidth * newHeight * channels);

    // For each pixel in the output image
    for (int y_out = 0; y_out < newHeight; ++y_out)
    {
        for (int x_out = 0; x_out < newWidth; ++x_out)
        {
            // Map the pixel to the input image
            float x_in = x_out / scale;
            float y_in = y_out / scale;

            // Find the coordinates of the four neighboring pixels
            int x0 = static_cast<int>(std::floor(x_in));
            int x1 = std::min(x0 + 1, width - 1);
            int y0 = static_cast<int>(std::floor(y_in));
            int y1 = std::min(y0 + 1, height - 1);

            // Calculate the distances between the neighboring pixels
            float dx = x_in - x0;
            float dy = y_in - y0;

            // Compute interpolation weights
            float w00 = (1 - dx) * (1 - dy);
            float w10 = dx * (1 - dy);
            float w01 = (1 - dx) * dy;
            float w11 = dx * dy;

            // For each color channel
            for (int c = 0; c < channels; ++c)
            {
                // Get the values of the four neighboring pixels
                uint8_t p00 = image[(y0 * width + x0) * channels + c];
                uint8_t p10 = image[(y0 * width + x1) * channels + c];
                uint8_t p01 = image[(y1 * width + x0) * channels + c];
                uint8_t p11 = image[(y1 * width + x1) * channels + c];

                // Compute the interpolated pixel value
                float value = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11;

                // Clamp the value between 0 and 255
                value = std::min(std::max(value, 0.0f), 255.0f);

                // Round the value to the nearest integer
                uint8_t interpolatedValue = static_cast<uint8_t>(std::round(value));

                // Set the pixel value in the output image
                outputImage[(y_out * newWidth + x_out) * channels + c] = interpolatedValue;
            }
        }
    }
    return outputImage;
}


// void send_axi_stream_input(const std::vector<uint8_t>& data) {
//     size_t data_index = 0;
//     bool done = false;

//     while (!done && main_time < sim_time) {
//         if (data_index < data.size()) {
//             // Prepare TDATA
//             uint64_t tdata_low = 0;
//             uint64_t tdata_high = 0;
//             int bytes_in_tdata = 16;  // Assuming TDATA is 128 bits (16 bytes)
//             uint8_t tdata_bytes[16] = {0};

//             // Load data into tdata_bytes
//             for (int i = 0; i < bytes_in_tdata && data_index < data.size(); ++i, ++data_index) {
//                 tdata_bytes[i] = data[data_index];
//             }

//             // Pack bytes into TDATA
//             TDATA_t tdata = pack_tdata(tdata_bytes);
//             top->image_r_TDATA = tdata.low;
//             // Assuming image_r_TDATA is 128 bits, adjust accordingly
//             top->image_r_TDATA |= (uint128_t(tdata.high) << 64);

//             // Assert TVALID
//             top->image_r_TVALID = 1;
//         } else {
//             // No more data to send
//             top->image_r_TVALID = 0;
//             done = true;
//         }

//         // Wait for TREADY
//         if (top->image_r_TVALID && top->image_r_TREADY) {
//             // Data has been accepted
//         }

//         tick();
//     }
// }

// std::vector<uint8_t> receive_axi_stream_output(size_t expected_size) {
//     bool done = false;
//     std::vector<uint8_t> received_data;
//     top->featureMap_TREADY = 1;  // Always ready to receive data

//     while (!done && main_time < sim_time) {
//         if (top->featureMap_TVALID && top->featureMap_TREADY) {
//             // Read TDATA
//             uint128_t tdata = top->featureMap_TDATA;
//             uint64_t tdata_low = tdata & 0xFFFFFFFFFFFFFFFFULL;
//             uint64_t tdata_high = (tdata >> 64) & 0xFFFFFFFFFFFFFFFFULL;
//             uint8_t tdata_bytes[16];

//             // Unpack TDATA into bytes
//             for (int i = 0; i < 8; ++i) {
//                 tdata_bytes[i] = (tdata_low >> (8 * i)) & 0xFF;
//                 tdata_bytes[i + 8] = (tdata_high >> (8 * i)) & 0xFF;
//             }

//             // Append bytes to received_data
//             for (int i = 0; i < 16; ++i) {
//                 received_data.push_back(tdata_bytes[i]);
//             }
//         }

//         // Check if we've received the expected amount of data
//         if (received_data.size() >= expected_size) {
//             done = true;
//         }
//         tick();
//     }
//     return received_data;
// }

// void monitor_and_compare_output(const std::vector<uint8_t>& expected_output, int& pass_count, int& fail_count, const std::vector<uint8_t>& input_data) {
//     std::vector<uint8_t> received_data;
//     size_t expected_size = expected_output.size();
//     received_data = receive_axi_stream_output(expected_size);

//     // Compare received data with expected output
//     bool passed = (received_data == expected_output);
//     if (passed) {
//         pass_count++;
//         std::cout << "Test PASSED!" << std::endl;
//     } 
//     else {
//         fail_count++;
//         std::cout << "Test FAILED!" << std::endl;
//         // Print input data
//         std::cout << "Input Data:\n" << std::endl;
//         for (auto val : input_data) std::cout << static_cast<int>(val) << " ";
//         std::cout << "\n" << std::endl;

//         // Print expected output
//         std::cout << "Expected Output:\n" << std::endl;
//         for (auto val : expected_output) std::cout << static_cast<int>(val) << " ";
//         std::cout << std::endl;

//         // Print received output
//         std::cout << "Received Output:\n" << std::endl;
//         for (auto val : received_data) std::cout << static_cast<int>(val) << " ";
//         std::cout << std::endl;

//         // Print differences
//         for (size_t i = 0; i < received_data.size(); ++i) {
//             if (received_data[i] != expected_output[i]) {
//                 std::cout << "Mismatch at index " << i << ": expected " << (int)expected_output[i]
//                           << ", got " << (int)received_data[i] << std::endl;
//             }
//         }
//     }
// }

// bool check_instantiation(VInterpolation_v1 *top) {

//     // Initial conditions
//     top->ap_clk = 0;
//     top->ap_rst_n = 0;
//     top->ap_start = 0;
//     top->image_r_TVALID = 0;
//     top->featureMap_TVALID = 0;
//     top->loadedInfo_empty_n = 0;

//     // Reset the module
//     for (int i = 0; i < 5; ++i) {
//         top->ap_clk = !top->ap_clk; // Toggle clock
//         top->eval();                // Evaluate the module
//         main_time += 5;             // Advance time
//     }
//     top->ap_rst_n = 1; // Release reset

//     // Check if module is idle
//     top->eval();
//     if (top->ap_idle) {
//         std::cout << "Module is idle after reset." << std::endl;
//     } else {
//         std::cout << "Module did not enter idle state after reset." << std::endl;
//         return false;
//     }

//     // Start the module
//     top->ap_start = 1;
//     for (int i = 0; i < 10; ++i) {
//         top->ap_clk = !top->ap_clk; // Toggle clock
//         top->eval();                // Evaluate the module
//         main_time += 5;
//     }
//     top->ap_start = 0; // Deassert start

//     // Check if module is ready
//     top->eval();
//     if (top->ap_ready) {
//         std::cout << "Module is ready to process data." << std::endl;
//         return true;
//     } else {
//         std::cout << "Module did not indicate readiness." << std::endl;
//         return false;
//     }
// }



// Helper function to compare two images
bool compareImages(const std::vector<uint8_t>& img1, const std::vector<uint8_t>& img2) {
    if (img1.size() != img2.size()) return false;
    for (size_t i = 0; i < img1.size(); ++i) {
        if (img1[i] != img2[i]) return false;
    }
    return true;
}

// Self-test function for multiple test cases to check bilinear interpolation function
bool runSelfTests() {
    std::vector<TestCase> testCases;
    // Test Case 1
    TestCase test1;
    test1.input_image.assign(input_test1, input_test1 + sizeof(input_test1) / sizeof(input_test1[0]));
    test1.expected_output.assign(output_test1, output_test1 + sizeof(output_test1) / sizeof(output_test1[0]));
    test1.height = 2;
    test1.width = 2;
    test1.new_width = 4;
    test1.new_height = 4;
    test1.channels = 3;
    testCases.push_back(test1);
    // Test Case 2
    TestCase test2;
    test2.input_image.assign(input_test2, input_test2 + sizeof(input_test2) / sizeof(input_test2[0]));
    test2.expected_output.assign(output_test2, output_test2 + sizeof(output_test2) / sizeof(output_test2[0]));
    test2.width = 2;
    test2.height = 2;
    test2.new_width = 4;
    test2.new_height = 4;
    test2.channels = 3;
    testCases.push_back(test2);
    // Test Case 3
    TestCase test3;
    test3.input_image.assign(input_test3, input_test3 + sizeof(input_test3) / sizeof(input_test3[0]));
    test3.expected_output.assign(output_test3, output_test3 + sizeof(output_test3) / sizeof(output_test3[0]));
    test3.width = 2;
    test3.height = 2;
    test3.new_width = 4;
    test3.new_height = 4;
    test3.channels = 3;
    testCases.push_back(test3);
    // Test Case 4
    TestCase test4;
    test4.input_image.assign(input_test4, input_test4 + sizeof(input_test4) / sizeof(input_test4[0]));
    test4.expected_output.assign(output_test4, output_test4 + sizeof(output_test4) / sizeof(output_test4[0]));
    test4.width = 2;
    test4.height = 2;
    test4.new_width = 4;
    test4.new_height = 4;
    test4.channels = 3;
    testCases.push_back(test4);
    // Test Case 5
    TestCase test5;
    test5.input_image.assign(input_test5, input_test5 + sizeof(input_test5) / sizeof(input_test5[0]));
    test5.expected_output.assign(output_test5, output_test5 + sizeof(output_test5) / sizeof(output_test5[0]));
    test5.width = 2;
    test5.height = 2;
    test5.new_width = 4;
    test5.new_height = 4;
    test5.channels = 3;
    testCases.push_back(test5);
    // Test Case 6
    TestCase test6;
    test6.input_image.assign(input_test6, input_test6 + sizeof(input_test6) / sizeof(input_test6[0]));
    test6.expected_output.assign(output_test6, output_test6 + sizeof(output_test6) / sizeof(output_test6[0]));
    test6.width = 28;
    test6.height = 28;
    test6.new_width = 56;
    test6.new_height = 56;
    test6.channels = 3;
    testCases.push_back(test6);

    bool allPassed = true;

    for (size_t i = 0; i < testCases.size(); ++i) {
        const auto& test = testCases[i];
        std::vector<uint8_t> result = bilinearInterpolation(test.input_image, test.width, test.height, test.channels,2);
        bool passed = compareImages(result, test.expected_output);

        // Report results
        std::cout << "Test Case " << (i + 1) << " - " << (passed ? "PASSED" : "FAILED") << std::endl;
        std::cout << "Input Image: \n";
        for (auto val : test.input_image) std::cout << static_cast<int>(val) << " ";
        std::cout << "\nExpected Output: \n";
        for (auto val : test.expected_output) std::cout << static_cast<int>(val) << " ";
        std::cout << "\nActual Output: \n";
        for (auto val : result) std::cout << static_cast<int>(val) << " ";
        std::cout << "\n" << std::endl;

        if (!passed) {
            allPassed = false;
        }
    }

    if (allPassed) {
        std::cout << "All test cases passed!" << std::endl;
        return true;
    } else {
        std::cout << "Some test cases failed. Check outputs for details." << std::endl;
        return false;
    }
}



int main(int argc, char **argv) {
    // Run self-tests for bilinear interpolation
    if (runSelfTests()){
        std::cout << "Self test passed." << std::endl;
    }
    else{
        std::cout << "Self test failed." << std::endl;
    }

    // Verilated::commandArgs(argc, argv); // Initialize Verilator
    // VInterpolation_v1 *top = new VInterpolation_v1; // Instantiate the module
    // bool is_ready = check_instantiation(top);
    // if (!is_ready) {
    //     std::cout << "Module instantiation failed. Exiting simulation." << std::endl;
    //     top->final();
    //     delete top;
    //     return 1;
    // }
    // single_image_test(top, 28, 28, 2.0, 255);

    // Verilated::commandArgs(argc, argv);

    // // Instantiate the module
    // top = new VInterpolation_v1;

    // // Enable waveform tracing
    // VerilatedVcdC* tfp = nullptr;
    // if (Verilated::commandArgsPlusMatch("trace")) {
    //     Verilated::traceEverOn(true);
    //     tfp = new VerilatedVcdC;
    //     top->trace(tfp, 99);
    //     tfp->open("waveform.vcd");
    // }

    // // Initialize the simulation
    // initialize();

    // // Number of test cases
    // int num_tests = 10;  // Adjust as needed
    // int pass_count = 0;
    // int fail_count = 0;

    // for (int test_num = 0; test_num < num_tests; ++test_num) {
    //     std::cout << "Running test case " << (test_num + 1) << "..." << std::endl;

    //     // Generate random input image
    //     int width = 2;
    //     int height = 2;
    //     int channels = 3;
    //     std::vector<uint8_t> input_data = generate_random_image(width, height, channels);

    //     // Compute expected output
    //     float scale = 2.0f;
    //     std::vector<uint8_t> expected_output = bilinearInterpolation(input_data, width, height, channels, scale);
    //     size_t expected_output_size = expected_output.size();

    //     // Reset the module before each test
    //     initialize();

    //     // Start the module
    //     start_module();

    //     // Send input data
    //     send_axi_stream_input(input_data);

    //     // Wait for the module to process data
    //     wait_for_module_done();

    //     // Monitor output and compare
    //     monitor_and_compare_output(expected_output, pass_count, fail_count, input_data);
    // }

    // // Print summary
    // std::cout << "Test Summary:" << std::endl;
    // std::cout << "Total tests run: " << num_tests << std::endl;
    // std::cout << "Passed: " << pass_count << std::endl;
    // std::cout << "Failed: " << fail_count << std::endl;

    // // Finalize simulation
    // top->final();
    // if (tfp) {
    //     tfp->close();
    //     delete tfp;
    // }
    // delete top;
    return 0;
}
