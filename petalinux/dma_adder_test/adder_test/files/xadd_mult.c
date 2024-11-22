// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.1 (64-bit)
// Tool Version Limit: 2023.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xadd_mult.h"

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

