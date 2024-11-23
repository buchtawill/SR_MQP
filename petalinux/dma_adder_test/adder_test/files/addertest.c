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
#define XADD_MULT_CTRL_BUS_ADDR_AP_CTRL         0x00
#define XADD_MULT_CTRL_BUS_ADDR_GIE             0x04
#define XADD_MULT_CTRL_BUS_ADDR_IER             0x08
#define XADD_MULT_CTRL_BUS_ADDR_ISR             0x0c
#define XADD_MULT_CTRL_BUS_ADDR_A_DATA          0x10
#define XADD_MULT_CTRL_BUS_BITS_A_DATA          32
#define XADD_MULT_CTRL_BUS_ADDR_B_DATA          0x18
#define XADD_MULT_CTRL_BUS_BITS_B_DATA          32
#define XADD_MULT_CTRL_BUS_ADDR_ADD_RESULT_DATA 0x20
#define XADD_MULT_CTRL_BUS_BITS_ADD_RESULT_DATA 32
#define XADD_MULT_CTRL_BUS_ADDR_ADD_RESULT_CTRL 0x24
#define XADD_MULT_CTRL_BUS_ADDR_SUB_RESULT_DATA 0x30
#define XADD_MULT_CTRL_BUS_BITS_SUB_RESULT_DATA 32
#define XADD_MULT_CTRL_BUS_ADDR_SUB_RESULT_CTRL 0x34
#define XADD_MULT_CTRL_BUS_ADDR_MUL_RESULT_DATA 0x40
#define XADD_MULT_CTRL_BUS_BITS_MUL_RESULT_DATA 32
#define XADD_MULT_CTRL_BUS_ADDR_MUL_RESULT_CTRL 0x44
#define XADD_MULT_CTRL_BUS_ADDR_DIV_RESULT_DATA 0x50
#define XADD_MULT_CTRL_BUS_BITS_DIV_RESULT_DATA 32
#define XADD_MULT_CTRL_BUS_ADDR_DIV_RESULT_CTRL 0x54

#ifndef XADD_MULT_H
#define XADD_MULT_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
    u16 DeviceId;
    u64 Ctrl_bus_BaseAddress;
} XAdd_mult_Config;
#endif

typedef struct {
    u64 Ctrl_bus_BaseAddress;
    u32 IsReady;
} XAdd_mult;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XAdd_mult_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XAdd_mult_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XAdd_mult_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XAdd_mult_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
int XAdd_mult_Initialize(XAdd_mult *InstancePtr, u16 DeviceId);
XAdd_mult_Config* XAdd_mult_LookupConfig(u16 DeviceId);
int XAdd_mult_CfgInitialize(XAdd_mult *InstancePtr, XAdd_mult_Config *ConfigPtr);
#else
int XAdd_mult_Initialize(XAdd_mult *InstancePtr, const char* InstanceName);
int XAdd_mult_Release(XAdd_mult *InstancePtr);
#endif

void XAdd_mult_Start(XAdd_mult *InstancePtr);
u32 XAdd_mult_IsDone(XAdd_mult *InstancePtr);
u32 XAdd_mult_IsIdle(XAdd_mult *InstancePtr);
u32 XAdd_mult_IsReady(XAdd_mult *InstancePtr);
void XAdd_mult_EnableAutoRestart(XAdd_mult *InstancePtr);
void XAdd_mult_DisableAutoRestart(XAdd_mult *InstancePtr);

void XAdd_mult_Set_a(XAdd_mult *InstancePtr, u32 Data);
u32 XAdd_mult_Get_a(XAdd_mult *InstancePtr);
void XAdd_mult_Set_b(XAdd_mult *InstancePtr, u32 Data);
u32 XAdd_mult_Get_b(XAdd_mult *InstancePtr);
u32 XAdd_mult_Get_add_result(XAdd_mult *InstancePtr);
u32 XAdd_mult_Get_add_result_vld(XAdd_mult *InstancePtr);
u32 XAdd_mult_Get_sub_result(XAdd_mult *InstancePtr);
u32 XAdd_mult_Get_sub_result_vld(XAdd_mult *InstancePtr);
u32 XAdd_mult_Get_mul_result(XAdd_mult *InstancePtr);
u32 XAdd_mult_Get_mul_result_vld(XAdd_mult *InstancePtr);
u32 XAdd_mult_Get_div_result(XAdd_mult *InstancePtr);
u32 XAdd_mult_Get_div_result_vld(XAdd_mult *InstancePtr);

void XAdd_mult_InterruptGlobalEnable(XAdd_mult *InstancePtr);
void XAdd_mult_InterruptGlobalDisable(XAdd_mult *InstancePtr);
void XAdd_mult_InterruptEnable(XAdd_mult *InstancePtr, u32 Mask);
void XAdd_mult_InterruptDisable(XAdd_mult *InstancePtr, u32 Mask);
void XAdd_mult_InterruptClear(XAdd_mult *InstancePtr, u32 Mask);
u32 XAdd_mult_InterruptGetEnabled(XAdd_mult *InstancePtr);
u32 XAdd_mult_InterruptGetStatus(XAdd_mult *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
//////////////////////////////////////////////////////////////// xadd_mult.h

#ifdef __linux__

/***************************** Include Files *********************************/
// #include "xadd_mult.h"

/***************** Macros (Inline Functions) Definitions *********************/
#define MAX_UIO_PATH_SIZE       256
#define MAX_UIO_NAME_SIZE       64
#define MAX_UIO_MAPS            5
#define UIO_INVALID_ADDR        0

/**************************** Type Definitions ******************************/
typedef struct {
    u64 addr;
    u32 size;
} XAdd_mult_uio_map;

typedef struct {
    int  uio_fd;
    int  uio_num;
    char name[ MAX_UIO_NAME_SIZE ];
    char version[ MAX_UIO_NAME_SIZE ];
    XAdd_mult_uio_map maps[ MAX_UIO_MAPS ];
} XAdd_mult_uio_info;

/***************** Variable Definitions **************************************/
static XAdd_mult_uio_info uio_info;

/************************** Function Implementation *************************/
static int line_from_file(char* filename, char* linebuf) {
    char* s;
    int i;
    FILE* fp = fopen(filename, "r");
    if (!fp) return -1;
    s = fgets(linebuf, MAX_UIO_NAME_SIZE, fp);
    fclose(fp);
    if (!s) return -2;
    for (i=0; (*s)&&(i<MAX_UIO_NAME_SIZE); i++) {
        if (*s == '\n') *s = 0;
        s++;
    }
    return 0;
}

static int uio_info_read_name(XAdd_mult_uio_info* info) {
    char file[ MAX_UIO_PATH_SIZE ];
    sprintf(file, "/sys/class/uio/uio%d/name", info->uio_num);
    return line_from_file(file, info->name);
}

static int uio_info_read_version(XAdd_mult_uio_info* info) {
    char file[ MAX_UIO_PATH_SIZE ];
    sprintf(file, "/sys/class/uio/uio%d/version", info->uio_num);
    return line_from_file(file, info->version);
}

static int uio_info_read_map_addr(XAdd_mult_uio_info* info, int n) {
    int ret;
    char file[ MAX_UIO_PATH_SIZE ];
    info->maps[n].addr = UIO_INVALID_ADDR;
    sprintf(file, "/sys/class/uio/uio%d/maps/map%d/addr", info->uio_num, n);
    FILE* fp = fopen(file, "r");
    if (!fp) return -1;
    ret = fscanf(fp, "0x%x", &info->maps[n].addr);
    fclose(fp);
    if (ret < 0) return -2;
    return 0;
}

static int uio_info_read_map_size(XAdd_mult_uio_info* info, int n) {
    int ret;
    char file[ MAX_UIO_PATH_SIZE ];
    sprintf(file, "/sys/class/uio/uio%d/maps/map%d/size", info->uio_num, n);
    FILE* fp = fopen(file, "r");
    if (!fp) return -1;
    ret = fscanf(fp, "0x%x", &info->maps[n].size);
    fclose(fp);
    if (ret < 0) return -2;
    return 0;
}

int XAdd_mult_Initialize(XAdd_mult *InstancePtr, const char* InstanceName) {
	XAdd_mult_uio_info *InfoPtr = &uio_info;
	struct dirent **namelist;
    int i, n;
    char* s;
    char file[ MAX_UIO_PATH_SIZE ];
    char name[ MAX_UIO_NAME_SIZE ];
    int flag = 0;

    assert(InstancePtr != NULL);

    n = scandir("/sys/class/uio", &namelist, 0, alphasort);
    if (n < 0)  return XST_DEVICE_NOT_FOUND;
    for (i = 0;  i < n; i++) {
    	strcpy(file, "/sys/class/uio/");
    	strcat(file, namelist[i]->d_name);
    	strcat(file, "/name");
        if ((line_from_file(file, name) == 0) && (strcmp(name, InstanceName) == 0)) {
            flag = 1;
            s = namelist[i]->d_name;
            s += 3; // "uio"
            InfoPtr->uio_num = atoi(s);
            break;
        }
    }
    if (flag == 0)  return XST_DEVICE_NOT_FOUND;

    uio_info_read_name(InfoPtr);
    uio_info_read_version(InfoPtr);
    for (n = 0; n < MAX_UIO_MAPS; ++n) {
        uio_info_read_map_addr(InfoPtr, n);
        uio_info_read_map_size(InfoPtr, n);
    }

    sprintf(file, "/dev/uio%d", InfoPtr->uio_num);
    if ((InfoPtr->uio_fd = open(file, O_RDWR)) < 0) {
        return XST_OPEN_DEVICE_FAILED;
    }

    // NOTE: slave interface 'Ctrl_bus' should be mapped to uioX/map0
    InstancePtr->Ctrl_bus_BaseAddress = (u64)mmap(NULL, InfoPtr->maps[0].size, PROT_READ|PROT_WRITE, MAP_SHARED, InfoPtr->uio_fd, 0 * getpagesize());
    assert(InstancePtr->Ctrl_bus_BaseAddress);

    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}

int XAdd_mult_Release(XAdd_mult *InstancePtr) {
	XAdd_mult_uio_info *InfoPtr = &uio_info;

    assert(InstancePtr != NULL);
    assert(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    munmap((void*)InstancePtr->Ctrl_bus_BaseAddress, InfoPtr->maps[0].size);

    close(InfoPtr->uio_fd);

    return XST_SUCCESS;
}

#endif
//////////////////////////////////////////////////////////////// end of xadd_mult_linux.c
/************************** Function Implementation *************************/
#ifndef __linux__
int XAdd_mult_CfgInitialize(XAdd_mult *InstancePtr, XAdd_mult_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Ctrl_bus_BaseAddress = ConfigPtr->Ctrl_bus_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XAdd_mult_Start(XAdd_mult *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XAdd_mult_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_AP_CTRL) & 0x80;
    XAdd_mult_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_AP_CTRL, Data | 0x01);
}

u32 XAdd_mult_IsDone(XAdd_mult *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XAdd_mult_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XAdd_mult_IsIdle(XAdd_mult *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XAdd_mult_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XAdd_mult_IsReady(XAdd_mult *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XAdd_mult_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XAdd_mult_EnableAutoRestart(XAdd_mult *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XAdd_mult_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_AP_CTRL, 0x80);
}

void XAdd_mult_DisableAutoRestart(XAdd_mult *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XAdd_mult_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_AP_CTRL, 0);
}

void XAdd_mult_Set_a(XAdd_mult *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XAdd_mult_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_A_DATA, Data);
}

u32 XAdd_mult_Get_a(XAdd_mult *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XAdd_mult_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_A_DATA);
    return Data;
}

void XAdd_mult_Set_b(XAdd_mult *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XAdd_mult_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_B_DATA, Data);
}

u32 XAdd_mult_Get_b(XAdd_mult *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XAdd_mult_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_B_DATA);
    return Data;
}

u32 XAdd_mult_Get_add_result(XAdd_mult *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XAdd_mult_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_ADD_RESULT_DATA);
    return Data;
}

u32 XAdd_mult_Get_add_result_vld(XAdd_mult *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XAdd_mult_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_ADD_RESULT_CTRL);
    return Data & 0x1;
}

u32 XAdd_mult_Get_sub_result(XAdd_mult *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XAdd_mult_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_SUB_RESULT_DATA);
    return Data;
}

u32 XAdd_mult_Get_sub_result_vld(XAdd_mult *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XAdd_mult_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_SUB_RESULT_CTRL);
    return Data & 0x1;
}

u32 XAdd_mult_Get_mul_result(XAdd_mult *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XAdd_mult_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_MUL_RESULT_DATA);
    return Data;
}

u32 XAdd_mult_Get_mul_result_vld(XAdd_mult *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XAdd_mult_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_MUL_RESULT_CTRL);
    return Data & 0x1;
}

u32 XAdd_mult_Get_div_result(XAdd_mult *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XAdd_mult_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_DIV_RESULT_DATA);
    return Data;
}

u32 XAdd_mult_Get_div_result_vld(XAdd_mult *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XAdd_mult_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_DIV_RESULT_CTRL);
    return Data & 0x1;
}

void XAdd_mult_InterruptGlobalEnable(XAdd_mult *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XAdd_mult_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_GIE, 1);
}

void XAdd_mult_InterruptGlobalDisable(XAdd_mult *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XAdd_mult_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_GIE, 0);
}

void XAdd_mult_InterruptEnable(XAdd_mult *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XAdd_mult_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_IER);
    XAdd_mult_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_IER, Register | Mask);
}

void XAdd_mult_InterruptDisable(XAdd_mult *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XAdd_mult_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_IER);
    XAdd_mult_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_IER, Register & (~Mask));
}

void XAdd_mult_InterruptClear(XAdd_mult *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XAdd_mult_WriteReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_ISR, Mask);
}

u32 XAdd_mult_InterruptGetEnabled(XAdd_mult *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XAdd_mult_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_IER);
}

u32 XAdd_mult_InterruptGetStatus(XAdd_mult *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XAdd_mult_ReadReg(InstancePtr->Ctrl_bus_BaseAddress, XADD_MULT_CTRL_BUS_ADDR_ISR);
}
//////////////////////////////////////////////////////////////// end of xadd_mult.c

// #include <stdio.h>
// #include <stdlib.h>  // for strtol
// #include "xadd_mult.h"

//Device name found in dma_no_bsp/components/plnx_workspace/device-tree/device-tree/pl.dtsi
#define DEVICE_NAME "add_mult_0"

// int main(int argc, char **argv)
// {
//     printf("INFO [addertest.c] Entering main()..\n\r");
//     XAdd_mult add_mult;

//     if(XAdd_mult_Initialize(&add_mult, DEVICE_NAME) != XST_SUCCESS){
//         printf("ERROR [addertest.c] Could not initialize add_mult\n\r");
//         return -1;
//     }

//     printf("INFO [addertest.c] Successfully initialized add_mult\n");

//     // printf("INFO [addertest.c] Add mult is ready: %d\n\r", XAdd_mult_IsReady(&add_mult));

//     printf("INFO [addertest.c] Setting a and b\n\r");
//     XAdd_mult_Set_a(&add_mult, 25);
//     XAdd_mult_Set_b(&add_mult, 10);

//     printf("INFO [addertest.c] Starting add_mult\n\r");

//     XAdd_mult_Start(&add_mult);

//     printf("INFO [addertest.c] Waiting for add_mult to finish...\n\r");
//     while (!XAdd_mult_IsDone(&add_mult)){}

//     printf("INFO [addertest.c] a+b result: %u\n\r", XAdd_mult_Get_add_result(&add_mult));
//     printf("INFO [addertest.c] a-b result: %u\n\r", XAdd_mult_Get_sub_result(&add_mult));
//     printf("INFO [addertest.c] a*b result: %u\n\r", XAdd_mult_Get_mul_result(&add_mult));
//     printf("INFO [addertest.c] a/b result: %u\n\r", XAdd_mult_Get_div_result(&add_mult));

//     return 0;
// }

#define ADD_MULT_BASE_ADDR  		0xA0000000
#define ADD_MULT_MEM_SIZE           0x10000

void set_a_mmap(uint32_t *add_mult_virtual_addr, uint32_t a){
    add_mult_virtual_addr[XADD_MULT_CTRL_BUS_ADDR_A_DATA>>2] = a;
}

void set_b_mmap(uint32_t *add_mult_virtual_addr, uint32_t b){
    add_mult_virtual_addr[XADD_MULT_CTRL_BUS_ADDR_B_DATA>>2] = b;
}

void start_adder_mmap(uint32_t *add_mult_virtual_addr){
    add_mult_virtual_addr[XADD_MULT_CTRL_BUS_ADDR_AP_CTRL>>2] = 0x01;
}

uint32_t is_done_mmap(uint32_t *add_mult_virtual_addr){
    return (add_mult_virtual_addr[XADD_MULT_CTRL_BUS_ADDR_AP_CTRL>>2] >> 1) & 0x1;
}

uint32_t is_idle_mmap(uint32_t *add_mult_virtual_addr){
    return (add_mult_virtual_addr[XADD_MULT_CTRL_BUS_ADDR_AP_CTRL>>2] >> 2) & 0x1;
}

uint32_t is_ready_mmap(uint32_t *add_mult_virtual_addr){
    return !(add_mult_virtual_addr[XADD_MULT_CTRL_BUS_ADDR_AP_CTRL>>2] & 0x1);
}

void enable_auto_restart_mmap(uint32_t *add_mult_virtual_addr){
    add_mult_virtual_addr[XADD_MULT_CTRL_BUS_ADDR_AP_CTRL>>2] = 0x80;
}

uint32_t get_add_result_mmap(uint32_t *add_mult_virtual_addr){
    return add_mult_virtual_addr[XADD_MULT_CTRL_BUS_ADDR_ADD_RESULT_DATA>>2];
}

uint32_t get_div_result_mmap(uint32_t *add_mult_virtual_addr){
    return add_mult_virtual_addr[XADD_MULT_CTRL_BUS_ADDR_DIV_RESULT_DATA>>2];
}

uint32_t get_mul_result_mmap(uint32_t *add_mult_virtual_addr){
    return add_mult_virtual_addr[XADD_MULT_CTRL_BUS_ADDR_MUL_RESULT_DATA>>2];
}

uint32_t get_sub_result_mmap(uint32_t *add_mult_virtual_addr){
    return add_mult_virtual_addr[XADD_MULT_CTRL_BUS_ADDR_SUB_RESULT_DATA>>2];
}

int main(int argc, char **argv){
    printf("INFO [addertest.c] Entering main()..\n");

    XAdd_mult add_mult;
    add_mult.Ctrl_bus_BaseAddress = ConfigPtr->Ctrl_bus_BaseAddress;
    add_mult.IsReady = XIL_COMPONENT_IS_READY;

    printf("INFO [addertest.c] Opening /dev/mem\n");
    int ddr_memory = open("/dev/mem", O_RDWR | O_SYNC);
    if(ddr_memory < 0){
        printf("ERROR [addertest.c] Failed to open /dev/mem: %s\n", strerror(errno));
        return -1;
    }

    printf("INFO [addertest.c] Memory mapping the address of the add_mult AXI IP via its AXI lite control interface\n");
    uint32_t *add_mult_virtual_addr = mmap(NULL, ADD_MULT_MEM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_memory, ADD_MULT_BASE_ADDR);

    if(add_mult_virtual_addr == MAP_FAILED){
        printf("ERROR [addertest.c] Failed to map add_mult AXI Lite register block: %s\n", strerror(errno));
        return -1;
    }

    printf("INFO [addertest.c] Setting a and b\n");
    set_a_mmap(add_mult_virtual_addr, 25);
    set_b_mmap(add_mult_virtual_addr, 10);
    start_adder_mmap(add_mult_virtual_addr);

    print("INFO [addertest.c] Waiting for add_mult to finish...\n");
    while (!is_done_mmap(add_mult_virtual_addr)){}

    printf("INFO [addertest.c] a+b result: %u\n", get_add_result_mmap(add_mult_virtual_addr));
    printf("INFO [addertest.c] a-b result: %u\n", get_sub_result_mmap(add_mult_virtual_addr));
    printf("INFO [addertest.c] a*b result: %u\n", get_mul_result_mmap(add_mult_virtual_addr));
    printf("INFO [addertest.c] a/b result: %u\n", get_div_result_mmap(add_mult_virtual_addr));

    munmap(add_mult_virtual_addr, ADD_MULT_MEM_SIZE);

    return 0;
}