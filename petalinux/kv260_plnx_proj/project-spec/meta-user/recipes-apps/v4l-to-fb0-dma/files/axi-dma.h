#ifndef AXI_DMA_H
#define AXI_DMA_H

#include "stdint.h"
#include "bits.h"

// Addresses found in SR_MQP/petalinux/kv260_plnx_proj/components/plnx_workspace/device-tree/device-tree/pl.dtsi
#define DMA_0_AXI_LITE_BASE			0xA0010000
#define DMA_1_AXI_LITE_BASE			0xA0020000
#define DMA_ADDRESS_SPACE_SIZE      0x00010000

#define KERNEL_RSVD_MEM_BASE		0x78000000
#define KERNEL_RSVD_MEM_SIZE        0x02000000

#define DMA_SELF_TEST_SRC_ADDR      KERNEL_RSVD_MEM_BASE   
#define DMA_SELF_TEST_DST_ADDR      (KERNEL_RSVD_MEM_BASE + 0x1000) 
#define DMA_SELF_TEST_LEN           0x1000

#define DMA_SYNC_TRIES				10000
#define MAX_DMA_SYNC_TRIES          0xFFFFFFFF

// Common names
#define MM2S_CONTROL_REGISTER       0x00
#define MM2S_STATUS_REGISTER        0x04
#define MM2S_SRC_ADDRESS_REGISTER   0x18
#define MM2S_TRNSFR_LENGTH_REGISTER 0x28

#define S2MM_CONTROL_REGISTER       0x30
#define S2MM_STATUS_REGISTER        0x34
#define S2MM_DST_ADDRESS_REGISTER   0x48
#define S2MM_BUFF_LENGTH_REGISTER   0x58

// Datasheet names
#define MM2S_DMACR                  0x00
#define MM2S_DMASR                  0x04
#define MM2S_SA_LSB32               0x18
#define MM2S_SA_MSB32               0x1C
#define MM2S_TRANSFER_LENGTH        0x28
#define S2MM_DMACR                  0x30
#define S2MM_DMASR                  0x34
#define S2MM_DA_LSB32               0x48
#define S2MM_DA_MSB32               0x4C
#define S2MM_TRANSFER_LENGTH        0x58

#define IOC_IRQ_FLAG                BIT12
#define IDLE_FLAG                   1<<1

#define STATUS_HALTED               0x00000001
#define STATUS_IDLE                 0x00000002
#define STATUS_SG_INCLDED           0x00000008
#define STATUS_DMA_INTERNAL_ERR     0x00000010
#define STATUS_DMA_SLAVE_ERR        0x00000020
#define STATUS_DMA_DECODE_ERR       0x00000040
#define STATUS_SG_INTERNAL_ERR      0x00000100
#define STATUS_SG_SLAVE_ERR         0x00000200
#define STATUS_SG_DECODE_ERR        0x00000400
#define STATUS_IOC_IRQ              0x00001000
#define STATUS_DELAY_IRQ            0x00002000
#define STATUS_ERR_IRQ              0x00004000

#define HALT_DMA                    0x00000000
#define RUN_DMA                     0x00000001
#define RESET_DMA                   0x00000004
#define ENABLE_IOC_IRQ              0x00001000
#define ENABLE_DELAY_IRQ            0x00002000
#define ENABLE_ERR_IRQ              0x00004000
#define ENABLE_ALL_IRQ              0x00007000


class AXIDMA {
private:
    uint32_t base_address;               // Base address of the AXI Lite port
    int mem_fd = -1;                     // File descriptor for /dev/mem
    volatile uint32_t *dma_phys_addr;    // Pointer after doing MMAP

    // Write to a DMA register
    void write_dma(uint32_t reg, uint32_t val);

public:

    /**
     * Constructor for the AXIDMA class
     * @param base_addr Base address of the AXI Lite port
     * @param dev_mem_fd File descriptor for /dev/mem
     * @return None
     */
    AXIDMA(uint32_t base_addr, int dev_mem_fd);
    ~AXIDMA();

    // Initialize the DMA by mmap'ing the file descriptor 
    // Return -1 for error during MMAP, -2 for other error, 0 for success
    int initialize();

    // Read from a DMA register
    uint32_t read_dma(uint32_t reg);

    /**
     * Read-Modify-Write to a DMA register
     * @param reg Register to read from
     * @param mask Mask to apply to the value
     * @param val Value to write to the register
     */
    void rmw_dma(uint32_t reg, uint32_t mask, uint32_t val);

    /**
     * Perform a self-test on the DMA engine by transferring data from a known memory location to the other
     * Need to call initialize before
     * @return 0 for success, -1 on timeout
     */
    int self_test();

    /**
     * Run the S2MM channel. 
     * @param dst_addr Destination address in memory space
     * @param len Length of the transfer in bytes
     * @param block True for blocking call, false for non-blocking
     * @return 0 for success, -1 on timeout
     */
    int transfer_s2mm(uint32_t dst_addr, uint32_t len, bool block);

    /**
     * Run the MM2S channel. This is a blocking call!
     * @param src_addr Source address in memory space
     * @param len Length of the transfer in bytes
     * @param block True for blocking call, false for non-blocking
     * @return 0 for success, -1 on timeout
     */
    int transfer_mm2s(uint32_t src_addr, uint32_t len, bool block);

    /**
     * Transfer len bytes from src_addr to dst_addr, running thru this DMA engine.
     * This is a blocking call!
     * @param src_addr Source address in memory space
     * @param dst_addr Destination address in memory space
     * @param len Length of the transfer in bytes
     * @param block True for blocking call, false for non-blocking
     * @return 0 for success, -1 on timeout
     */
    int transfer(uint32_t src_addr, uint32_t dst_addr, uint32_t len, bool block = true);
    
    /**
     * Get the base address of the AXI Lite port
     * @return Base address of the AXI Lite port
     */
    uint32_t getBaseAddress() const {
        return base_address;
    }

    /**
     * Set the base address of the AXI Lite port
     * @param base_addr Base address of the AXI Lite port
     */
    void setBaseAddress(uint32_t base_addr) {
        base_address = base_addr;
    }

    /**
     * Wait for an IOC flag on either MM2S_DMASR or S2MM_DMASR
     * @param reg Register to read from - MM2S_DMASR or S2MM_DMASR
     * @param max_tries Maximum number of tries before timeout. Default is DMA_SYNC_TRIES
     * @return Number of reg reads on success, -1 on timeout
     */
    int sync_channel(uint32_t channel_status_reg, uint32_t max_tries = DMA_SYNC_TRIES);

    /**
     * Set the length of the MM2S outbound transfer
     * @param transfer_length Length of the transfer in bytes
     * @return None
     */
    void set_mm2s_len(uint16_t transfer_length);

    /**
     * Set the length of the S2MM inbound transfer
     * @param transfer_length Length of the transfer in bytes
     * @return None
     */
    void set_s2mm_len(uint16_t transfer_length);

    /**
     * Set the source address for the MM2S outbound transfer
     * @param source Source address in memory space
     * @return None
     */
    void set_mm2s_src(uint32_t source);

    /**
     * Set the destination address for the S2MM inbound transfer
     * @param destination Destination address in memory space
     * @return None
     */
    void set_s2mm_dest(uint32_t destination);

    /**
     * Reset the MM2S channel
     * @return None
     */
    void reset_mm2s();

    /**
     * Reset the S2MM channel
     */
    void reset_s2mm();

    /**
     * Start the MM2S channel
     */
    void start_mm2s();

    /**
     * Start the S2MM channel
     */
    void start_s2mm();

    /**
     * Halt the MM2S channel
     */
    void halt_mm2s();

    /**
     * Halt the S2MM channel
     */
    void halt_s2mm();

    /**
     * Print info about a status register
     * @param reg either MM2S_DMASR or S2MM_DMASR
     */
    void print_status(uint32_t reg);

    /**
     * Reset both DMA channels
     */
    void reset_dma(){
        reset_mm2s();
        reset_s2mm();
    }

    /**
     * Halt both DMA channels
     */
    void halt_dma(){
        halt_mm2s();
        halt_s2mm();
    }

    void enable_s2mm_intr();
    void enable_mm2s_intr();

    void enable_all_intr(){
        enable_s2mm_intr();
        enable_mm2s_intr();
    }

    /**
     * Clear the IOC bit in a status register
     * @param reg Register to clear the IOC bit from. Either MM2S_DMASR or S2MM_DMASR
     * @return None
     */
    void clear_irq_bit(uint32_t reg){
        rmw_dma(reg, IOC_IRQ_FLAG, IOC_IRQ_FLAG);
    }

    /**
     * Clear the IOC bit in both status registers
     * @return None
     */
    void clear_irq_bits(){
        rmw_dma(MM2S_DMASR, IOC_IRQ_FLAG, IOC_IRQ_FLAG);
        rmw_dma(S2MM_DMASR, IOC_IRQ_FLAG, IOC_IRQ_FLAG);
    }
};

#endif // AXI_DMA_H