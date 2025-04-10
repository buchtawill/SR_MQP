// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.1 (64-bit)
// Tool Version Limit: 2023.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
// control
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read/COR)
//        bit 7  - auto_restart (Read/Write)
//        bit 9  - interrupt (Read)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0 - enable ap_done interrupt (Read/Write)
//        bit 1 - enable ap_ready interrupt (Read/Write)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0 - ap_done (Read/TOW)
//        bit 1 - ap_ready (Read/TOW)
//        others - reserved
// 0x10 : Data signal of threshold
//        bit 31~0 - threshold[31:0] (Read/Write)
// 0x14 : reserved
// 0x18 : Data signal of override_mode
//        bit 1~0 - override_mode[1:0] (Read/Write)
//        others  - reserved
// 0x1c : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XPROCESS_TILE_CONTROL_ADDR_AP_CTRL            0x00
#define XPROCESS_TILE_CONTROL_ADDR_GIE                0x04
#define XPROCESS_TILE_CONTROL_ADDR_IER                0x08
#define XPROCESS_TILE_CONTROL_ADDR_ISR                0x0c
#define XPROCESS_TILE_CONTROL_ADDR_THRESHOLD_DATA     0x10
#define XPROCESS_TILE_CONTROL_BITS_THRESHOLD_DATA     32
#define XPROCESS_TILE_CONTROL_ADDR_OVERRIDE_MODE_DATA 0x18
#define XPROCESS_TILE_CONTROL_BITS_OVERRIDE_MODE_DATA 2

