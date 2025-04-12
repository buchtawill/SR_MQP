// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.1 (64-bit)
// Tool Version Limit: 2023.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XPROCESS_TILE_H
#define XPROCESS_TILE_H

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
#include "xprocess_tile_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
    u16 DeviceId;
    u64 Control_BaseAddress;
} XProcess_tile_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XProcess_tile;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XProcess_tile_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XProcess_tile_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XProcess_tile_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XProcess_tile_ReadReg(BaseAddress, RegOffset) \
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
int XProcess_tile_Initialize(XProcess_tile *InstancePtr, u16 DeviceId);
XProcess_tile_Config* XProcess_tile_LookupConfig(u16 DeviceId);
int XProcess_tile_CfgInitialize(XProcess_tile *InstancePtr, XProcess_tile_Config *ConfigPtr);
#else
int XProcess_tile_Initialize(XProcess_tile *InstancePtr, const char* InstanceName);
int XProcess_tile_Release(XProcess_tile *InstancePtr);
#endif

void XProcess_tile_Start(XProcess_tile *InstancePtr);
u32 XProcess_tile_IsDone(XProcess_tile *InstancePtr);
u32 XProcess_tile_IsIdle(XProcess_tile *InstancePtr);
u32 XProcess_tile_IsReady(XProcess_tile *InstancePtr);
void XProcess_tile_EnableAutoRestart(XProcess_tile *InstancePtr);
void XProcess_tile_DisableAutoRestart(XProcess_tile *InstancePtr);

void XProcess_tile_Set_threshold(XProcess_tile *InstancePtr, u32 Data);
u32 XProcess_tile_Get_threshold(XProcess_tile *InstancePtr);
void XProcess_tile_Set_override_mode(XProcess_tile *InstancePtr, u32 Data);
u32 XProcess_tile_Get_override_mode(XProcess_tile *InstancePtr);

void XProcess_tile_InterruptGlobalEnable(XProcess_tile *InstancePtr);
void XProcess_tile_InterruptGlobalDisable(XProcess_tile *InstancePtr);
void XProcess_tile_InterruptEnable(XProcess_tile *InstancePtr, u32 Mask);
void XProcess_tile_InterruptDisable(XProcess_tile *InstancePtr, u32 Mask);
void XProcess_tile_InterruptClear(XProcess_tile *InstancePtr, u32 Mask);
u32 XProcess_tile_InterruptGetEnabled(XProcess_tile *InstancePtr);
u32 XProcess_tile_InterruptGetStatus(XProcess_tile *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
