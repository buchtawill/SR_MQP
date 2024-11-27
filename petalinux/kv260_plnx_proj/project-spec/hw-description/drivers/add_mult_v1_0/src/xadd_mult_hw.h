// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2023.1 (64-bit)
// Tool Version Limit: 2023.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
// CTRL_BUS
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
// 0x10 : Data signal of a
//        bit 31~0 - a[31:0] (Read/Write)
// 0x14 : reserved
// 0x18 : Data signal of b
//        bit 31~0 - b[31:0] (Read/Write)
// 0x1c : reserved
// 0x20 : Data signal of add_result
//        bit 31~0 - add_result[31:0] (Read)
// 0x24 : Control signal of add_result
//        bit 0  - add_result_ap_vld (Read/COR)
//        others - reserved
// 0x30 : Data signal of sub_result
//        bit 31~0 - sub_result[31:0] (Read)
// 0x34 : Control signal of sub_result
//        bit 0  - sub_result_ap_vld (Read/COR)
//        others - reserved
// 0x40 : Data signal of mul_result
//        bit 31~0 - mul_result[31:0] (Read)
// 0x44 : Control signal of mul_result
//        bit 0  - mul_result_ap_vld (Read/COR)
//        others - reserved
// 0x50 : Data signal of div_result
//        bit 31~0 - div_result[31:0] (Read)
// 0x54 : Control signal of div_result
//        bit 0  - div_result_ap_vld (Read/COR)
//        others - reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

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

