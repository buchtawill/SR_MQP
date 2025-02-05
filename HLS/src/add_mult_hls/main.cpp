#include <ap_int.h>

void add_mult(ap_uint<32> a, ap_uint<32> b, ap_uint<32> &add_result, ap_uint<32> &sub_result, ap_uint<32> &mul_result, ap_uint<32> &div_result) {
    #pragma HLS INTERFACE s_axilite port=a bundle=CTRL_BUS
    #pragma HLS INTERFACE s_axilite port=b bundle=CTRL_BUS
    #pragma HLS INTERFACE s_axilite port=add_result bundle=CTRL_BUS
    #pragma HLS INTERFACE s_axilite port=sub_result bundle=CTRL_BUS
    #pragma HLS INTERFACE s_axilite port=mul_result bundle=CTRL_BUS
    #pragma HLS INTERFACE s_axilite port=div_result bundle=CTRL_BUS
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS

    // Perform arithmetic operations
    add_result = a + b;
    sub_result = a - b;
    mul_result = a * b;
    //div_result = (b != 0) ? (a / b) : 0;  // Division with check for divide by zero
    if(b == 0) div_result = 0;
    else div_result = a / b;
}
