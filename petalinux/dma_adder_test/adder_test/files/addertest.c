/*
* Copyright (C) 2013-2022  Xilinx, Inc.  All rights reserved.
* Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
*
* Permission is hereby granted, free of charge, to any person
* obtaining a copy of this software and associated documentation
* files (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL XILINX  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
* CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
* Except as contained in this notice, the name of the Xilinx shall not be used
* in advertising or otherwise to promote the sale, use or other dealings in this
* Software without prior written authorization from Xilinx.
*
*/

#include <stdio.h>
// #include <stdlib.h>  // for strtol
#include "xadd_mult.h"

//Device name found in dma_no_bsp/components/plnx_workspace/device-tree/device-tree/pl.dtsi
#define DEVICE_NAME "add_mult_0"

int main(int argc, char **argv)
{
    printf("INFO [addertest.c] Entering main()\n\r");
    XAdd_mult add_mult;

    XAdd_mult_Initialize(&add_mult, DEVICE_NAME);

    printf("INFO [addertest.c] Add mult is ready: %d\n\r", XAdd_mult_IsReady(&add_mult));

    printf("INFO [addertest.c] Setting a and b\n\r");
    XAdd_mult_Set_a(&add_mult, 25);
    XAdd_mult_Set_b(&add_mult, 10);

    XAdd_mult_Start(&add_mult);

    printf("INFO [addertest.c] Waiting for add_mult to finish...\n\r");
    while (!XAdd_mult_IsDone(&add_mult)){}

    printf("INFO [addertest.c] a+b result: %u\n\r", XAdd_mult_Get_add_result(&add_mult));
    printf("INFO [addertest.c] a-b result: %u\n\r", XAdd_mult_Get_sub_result(&add_mult));
    printf("INFO [addertest.c] a*b result: %u\n\r", XAdd_mult_Get_mul_result(&add_mult));
    printf("INFO [addertest.c] a/b result: %u\n\r", XAdd_mult_Get_div_result(&add_mult));

    return 0;
}
