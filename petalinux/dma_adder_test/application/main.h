#ifndef MAIN_H_
#define MAIN_H_

#define ADD_MULT_BASE_ADDR  0xA0000000
#define ADD_MULT_REG_(i)    (ADD_MULT_BASE_ADDR + 4*i)

#define DATA_A_OFFSET       0x10
#define DATA_B_OFFSET       0x18

#define ADD_RES_OFFSET      0x20
#define SUB_RES_OFFSET      0x30
#define MUL_RES_OFFSET      0x40
#define DIV_RES_OFFSET      0x50
//------------------------Address Info-------------------
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


#endif //MAIN_H_