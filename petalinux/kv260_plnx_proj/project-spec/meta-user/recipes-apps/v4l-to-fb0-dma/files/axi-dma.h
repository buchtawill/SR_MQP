#ifndef AXI_DMA_H
#define AXI_DMA_H

#include "stdint.h"

#define DMA_0_AXI_LITE_BASE			0xA0010000
#define DMA_1_AXI_LITE_BASE			0xA0030000
#define DMA_ADDRESS_SPACE_SIZE      0x00010000

#define VIRTUAL_SRC_ADDR 			0x78000000
#define VIRTUAL_DST_ADDR 			0x79000000

#define DMA_SYNC_TRIES				100
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

#define IOC_IRQ_FLAG                1<<12
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
    uint32_t base_address;          // Base address of the AXI Lite port
    int mem_fd = -1;                // File descriptor for /dev/mem
    uint32_t *dma_phys_addr = nullptr; // Pointer after doing MMAP

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
     * @param max_tries Maximum number of tries before timeout
     * @return Number of reg reads on success, -1 on timeout
     */
    int wait_for_channel_completion(uint32_t channel_status_reg, uint32_t max_tries = DMA_SYNC_TRIES);

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
     * Print the status of the MM2S channel
     */
    void print_s2mm_status();

    /**
     * Print the status of the S2MM channel
     */
    void print_mm2s_status();

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
};

#endif // AXI_DMA_H