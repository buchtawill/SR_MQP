#ifndef AXI_DMA_H
#define AXI_DMA_H

#include "bits.h"
#include "dma-sg-bd.h"
#include <stdint.h>

// #define DMA_SG_MODE 1
// #define DMA_DIRECT_REG_MODE 1

// Default to direct register mode
#ifndef DMA_SG_MODE
#define DMA_DIRECT_REG_MODE 1
#pragma message("INFO [axi-dma.h] AXI DMA is in Direct Register mode")
#else
#pragma message("INFO [axi-dma.h] AXI DMA is in Scatter Gather mode")
#endif

// Configured in Vivado
#define DMA_ADDRESS_SPACE_SIZE      0x00010000

// Length of DMA self test transfer in bytes
#define DMA_SELF_TEST_LEN           ((uint32_t)4096)

#define DMA_SYNC_TRIES				10000
#define MAX_DMA_SYNC_TRIES          0xFFFFFFFF

#ifndef DMA_SG_MODE
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

#else  // Scatter gather mode

// Datasheet names
#define MM2S_DMACR                  0x00 // MM2S Control register
#define MM2S_DMASR                  0x04 // MM2S Status register
#define MM2S_CURDESC                0x08 // MM2S Current descriptor pointer
#define MM2S_CURDESC_MSB            0x0C // MM2S Current descriptor pointer, MSB
#define MM2S_TAILDESC               0x10 // MM2S Tail descriptor pointer
#define MM2S_TAILDESC_MSB           0x14 // MM2S Tail descriptor pointer, MSB

#define SG_CTL_REG                  0x2C // Only available when DMA is in multi channel mode

#define S2MM_DMACR                  0x30 // S2MM Control register
#define S2MM_DMASR                  0x34 // S2MM Status register
#define S2MM_CURDESC                0x38 // S2MM Current descriptor pointer
#define S2MM_CURDESC_MSB            0x3C // S2MM Current descriptor pointer, MSB
#define S2MM_TAILDESC               0x40 // S2MM Tail descriptor pointer
#define S2MM_TAILDESC_MSB           0x44 // S2MM Tail descriptor pointer, MSB

// TODO: Make this be relative, so that it can be used in other projects
// #define DMA_BD_MEM_BASE             KERNEL_RSVD_MEM_BASE + 0x01000000
#define DMA_BD_MEM_BASE             0x79000000
#define DMA_BD_MEM_SIZE_BYTES       ((uint32_t)0x00080000)              // 512kB of space for BD rings

#define DMA_BDR_S2MM_BASE           DMA_BD_MEM_BASE
#define DMA_BDR_S2MM_SIZE_BYTES     ((uint32_t)0x00040000)
#define DMA_BDR_MM2S_BASE           (DMA_BD_MEM_BASE + DMA_BDR_S2MM_SIZE_BYTES)      // 256kB for MM2S, 256kB for S2MM
#define DMA_BDR_MM2S_SIZE_BYTES     ((uint32_t)0x00040000)

#define NUM_BD_PER_CHANNEL          4096 // DMA_BDR_MM2S_SIZE_BYTES / sizeof(DMA_SG_BD)

#endif // DMA_SG_MODE

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
    uint32_t base_address;              // Base address of the AXI Lite port
    uint32_t total_bytes_mm2s = 0;      // Total bytes transferred in MM2S channel
    uint32_t total_bytes_s2mm = 0;      // Total bytes transferred in S2MM channel
    uint32_t n_s2mm_calls = 0;
    uint32_t n_mm2s_calls = 0;
    int mem_fd = -1;                    // File descriptor for /dev/mem
    volatile uint32_t *dma_phys_addr;   // Pointer after doing MMAP to AXI Lite base

    #ifdef DMA_SG_MODE
    volatile DMA_SG_BD *mm2s_bd_arr;     // Scatter Gather buffer descriptor for MM2S
    volatile DMA_SG_BD *s2mm_bd_arr;     // Scatter Gather buffer descriptor for S2MM
    volatile DMA_SG_BD *s2mm_tail;
    volatile DMA_SG_BD *mm2s_tail;
    uint32_t mm2s_tail_address;
    uint32_t s2mm_tail_address;
    #endif

    // Write to a DMA register
    void write_dma(uint32_t reg, uint32_t val);

    /**
     * Self test for the DMA engine in Scatter Gather mode
     * @return 0 for success, -1 on failure
     */
    int self_test_sg();

    /**
     * Self test for the DMA engine in Direct Register mode
     * @return 0 for success, -1 on failure
     */
    int self_test_dr();

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
     * Print some debug info about the DMA core
     */
    void print_debug_info();

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

    #ifndef DMA_SG_MODE
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

    #else // Scatter gather mode

    /**
     * Get a pointer to a buffer descriptor in the MM2S BD ring
     * @param index Index of the buffer descriptor
     * @return Pointer to the buffer descriptor, or nullptr if index is out of bounds
     */
    volatile DMA_SG_BD* get_mm2s_bd(int index){
        if(index < 0 || index >= NUM_BD_PER_CHANNEL){
            return nullptr;
        }
        return &mm2s_bd_arr[index];
    }

    /**
     * Get a pointer to a buffer descriptor in the S2MM BD ring
     * @param index Index of the buffer descriptor
     * @return Pointer to the buffer descriptor, or nullptr if index is out of bounds
     */
    volatile DMA_SG_BD* get_s2mm_bd(int index){
        if(index < 0 || index >= NUM_BD_PER_CHANNEL){
            return nullptr;
        }
        return &s2mm_bd_arr[index];
    }

    /**
     * Reset the s2mm BD ring, creating a ring that has num_bds buffer descriptors. 
     * TODO: Right now, this resets all BD memory. In the future, have it return a new set of buffer descriptors
     * The ring will start at index 0 and end at index num_bds - 1
     * This function will set the next desc addresses and RXSOF/RXEOF appropriately. 
     */
    void create_s2mm_bd_ring(int num_bds);

    /**
     * Reset the mm2s BD ring, creating a ring that has num_bds buffer descriptors. 
     * TODO: Right now, this resets all BD memory. In the future, have it return a new set of buffer descriptors
     * The ring will start at index 0 and end at index num_bds - 1
     * This function will set the next desc addresses and SOF/EOF appropriately. 
     */
    void create_mm2s_bd_ring(int num_bds);

    /**
     * Start a transfer using both the MM2S and S2MM BD rings. Block until complete or timeout
     * This function assumes mm2s_tail_address and s2mm_tail_address are already set in this class
     * @return 0 on success, -1 on error
     */
    int transfer_sg();

    /**
     * Set the buffer address for a buffer descriptor in the MM2S BD ring
     * @param idx Index of the buffer descriptor
     * @param addr Address to set in the buffer descriptor
     * @return None
     */
    void set_mm2s_bd_buff_addr(int idx, uint32_t addr){
        this->mm2s_bd_arr[idx].buffer_address = addr;
    }

    /**
     * Set the buffer address for a buffer descriptor in the S2MM BD ring
     * @param idx Index of the buffer descriptor
     * @param addr Address to set in the buffer descriptor
     * @return None
     */
    void set_s2mm_bd_buff_addr(int idx, uint32_t addr){
        this->s2mm_bd_arr[idx].buffer_address = addr;
    }

    /**
     * Set the buffer length for a buffer descriptor in the MM2S BD ring
     * @param idx Index of the buffer descriptor
     * @param len Length to set in the buffer descriptor
     * @return None
     */
    void set_mm2s_bd_len(int idx, uint32_t len){
        set_buffer_length(&mm2s_bd_arr[idx], len);
    }

    /**
     * Set the buffer length for a buffer descriptor in the S2MM BD ring
     * @param idx Index of the buffer descriptor
     * @param len Length to set in the buffer descriptor
     * @return None
     */
    void set_s2mm_bd_len(int idx, uint32_t len){
        set_buffer_length(&s2mm_bd_arr[idx], len);
    }

    #endif // Scatter gather mode

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
