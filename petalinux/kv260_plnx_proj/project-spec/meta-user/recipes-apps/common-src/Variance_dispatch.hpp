#ifndef VARIANCE_DISPATCH_H
#define VARIANCE_DISPATCH_H

#include <stdint-gcc.h>
#include "phys-mman.h"
#include "PhysMem.h"

// Variance block class is going to need:
// - Base address
// - Function to start
// - Function to read if done
// - Function to read if ready

/**
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
 */

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

#define VARIARNCE_OVERRIDE_MODE_DEFAULT (uint32_t)0
#define VARIARNCE_OVERRIDE_MODE_CONV (uint32_t)1
#define VARIARNCE_OVERRIDE_MODE_INTERP (uint32_t)2

class VarianceDispatcher{
private:
    uint32_t base_address_phys;
    uint32_t dev_mem_fd;
    PhysMem *mem_block;

public:
    VarianceDispatcher(uint32_t base_addr, uint32_t dev_mem_fd);
    ~VarianceDispatcher();

    /**
     * Return 0 on success, -1 on error
     */
    int init();

    void start();
    uint32_t is_done();
    uint32_t is_idle();
    uint32_t is_ready();
    void enable_auto_restart();
    void disable_auto_restart();

    void set_threshold(uint32_t value);
    uint32_t get_threshold();

    void set_override(uint32_t mode);
    uint32_t get_override();

};

#endif
