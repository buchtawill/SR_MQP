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
#include "xprocess_tile.h"

extern XProcess_tile_Config XProcess_tile_ConfigTable[];

XProcess_tile_Config *XProcess_tile_LookupConfig(u16 DeviceId) {
	XProcess_tile_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XPROCESS_TILE_NUM_INSTANCES; Index++) {
		if (XProcess_tile_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XProcess_tile_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XProcess_tile_Initialize(XProcess_tile *InstancePtr, u16 DeviceId) {
	XProcess_tile_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XProcess_tile_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XProcess_tile_CfgInitialize(InstancePtr, ConfigPtr);
}

#endif

