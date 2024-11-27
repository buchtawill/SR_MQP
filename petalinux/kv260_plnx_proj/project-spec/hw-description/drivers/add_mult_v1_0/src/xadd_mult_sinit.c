// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.1 (64-bit)
// Tool Version Limit: 2023.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#include "xparameters.h"
#include "xadd_mult.h"

extern XAdd_mult_Config XAdd_mult_ConfigTable[];

XAdd_mult_Config *XAdd_mult_LookupConfig(u16 DeviceId) {
	XAdd_mult_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XADD_MULT_NUM_INSTANCES; Index++) {
		if (XAdd_mult_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XAdd_mult_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XAdd_mult_Initialize(XAdd_mult *InstancePtr, u16 DeviceId) {
	XAdd_mult_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XAdd_mult_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XAdd_mult_CfgInitialize(InstancePtr, ConfigPtr);
}

#endif

