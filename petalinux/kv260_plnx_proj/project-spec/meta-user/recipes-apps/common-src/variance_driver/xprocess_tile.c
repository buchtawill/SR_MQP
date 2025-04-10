// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.1 (64-bit)
// Tool Version Limit: 2023.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xprocess_tile.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XProcess_tile_CfgInitialize(XProcess_tile *InstancePtr, XProcess_tile_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XProcess_tile_Start(XProcess_tile *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XProcess_tile_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_TILE_CONTROL_ADDR_AP_CTRL) & 0x80;
    XProcess_tile_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_TILE_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XProcess_tile_IsDone(XProcess_tile *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XProcess_tile_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_TILE_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XProcess_tile_IsIdle(XProcess_tile *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XProcess_tile_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_TILE_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XProcess_tile_IsReady(XProcess_tile *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XProcess_tile_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_TILE_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XProcess_tile_EnableAutoRestart(XProcess_tile *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XProcess_tile_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_TILE_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XProcess_tile_DisableAutoRestart(XProcess_tile *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XProcess_tile_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_TILE_CONTROL_ADDR_AP_CTRL, 0);
}

void XProcess_tile_Set_threshold(XProcess_tile *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XProcess_tile_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_TILE_CONTROL_ADDR_THRESHOLD_DATA, Data);
}

u32 XProcess_tile_Get_threshold(XProcess_tile *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XProcess_tile_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_TILE_CONTROL_ADDR_THRESHOLD_DATA);
    return Data;
}

void XProcess_tile_Set_override_mode(XProcess_tile *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XProcess_tile_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_TILE_CONTROL_ADDR_OVERRIDE_MODE_DATA, Data);
}

u32 XProcess_tile_Get_override_mode(XProcess_tile *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XProcess_tile_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_TILE_CONTROL_ADDR_OVERRIDE_MODE_DATA);
    return Data;
}

void XProcess_tile_InterruptGlobalEnable(XProcess_tile *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XProcess_tile_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_TILE_CONTROL_ADDR_GIE, 1);
}

void XProcess_tile_InterruptGlobalDisable(XProcess_tile *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XProcess_tile_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_TILE_CONTROL_ADDR_GIE, 0);
}

void XProcess_tile_InterruptEnable(XProcess_tile *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XProcess_tile_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_TILE_CONTROL_ADDR_IER);
    XProcess_tile_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_TILE_CONTROL_ADDR_IER, Register | Mask);
}

void XProcess_tile_InterruptDisable(XProcess_tile *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XProcess_tile_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_TILE_CONTROL_ADDR_IER);
    XProcess_tile_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_TILE_CONTROL_ADDR_IER, Register & (~Mask));
}

void XProcess_tile_InterruptClear(XProcess_tile *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XProcess_tile_WriteReg(InstancePtr->Control_BaseAddress, XPROCESS_TILE_CONTROL_ADDR_ISR, Mask);
}

u32 XProcess_tile_InterruptGetEnabled(XProcess_tile *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XProcess_tile_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_TILE_CONTROL_ADDR_IER);
}

u32 XProcess_tile_InterruptGetStatus(XProcess_tile *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XProcess_tile_ReadReg(InstancePtr->Control_BaseAddress, XPROCESS_TILE_CONTROL_ADDR_ISR);
}

