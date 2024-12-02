// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.1 (64-bit)
// Tool Version Limit: 2023.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
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
#include "xadd_mult_hw.h"

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
