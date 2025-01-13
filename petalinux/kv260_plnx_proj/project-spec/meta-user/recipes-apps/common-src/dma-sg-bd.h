#ifndef DMA_SG_BD_H_
#define DMA_SG_BD_H_

#include <stdint.h>
#include "axi-dma.h"


#define BD_CONTROL_BUF_LEN_MASK 0x03ffffff

#define SG_BD_SIZE_BYTES 64

// See Scatter Gather descriptor table in PG021. Starts at page 36 of v7.1
// Buffer descriptors must be in memory that is accessible by the DMA engine,
// and must be aligned to a 16-word boundary (64 bytes).
typedef struct{
    uint32_t next_desc_ptr;         // Next descriptor pointer 
    uint32_t next_desc_ptr_msb;     // Next descriptor pointer, MSB (unused)
    uint32_t buffer_address;        // Buffer address
    uint32_t buffer_address_msb;    // Buffer address, MSB (unused)
    uint32_t rsvd1;                 // Placeholder to offset memory
    uint32_t rsvd2;                 // Placeholder to offset memory
    uint32_t control;               // Control
    uint32_t status;                // Status
    uint32_t app[5];                // Application specific
    uint32_t rsvd[2];               // Reserved to make the whole struct 64 bytes and align the next descriptor
    uint32_t next_desc_index;       // Index of the next descriptor in the ring
} DMA_SG_BD;

typedef volatile DMA_SG_BD * BD_PTR;

// /**
//  * Set the next descriptor pointer reg in the buffer descriptor
//  * @param bd Pointer to the buffer descriptor
//  * @param next_desc_ptr Next descriptor pointer
//  * @return None
//  */
// void set_bd_next_desc_ptr(BD_PTR bd, uint32_t next_desc_ptr);

// // void set_bd_nxt_idx_and_desc_ptr(BD_PTR bd, uint32_t index);

// /**
//  * Set the buffer address reg in the buffer descriptor
//  * @param bd Pointer to the buffer descriptor
//  * @param buffer_address Buffer address
//  * @return None
//  */
// void set_bd_buffer_address(BD_PTR bd, uint32_t buffer_address);

// /**
//  * Set the control reg in the buffer descriptor
//  * @param bd Pointer to the buffer descriptor
//  * @param control Control value
//  * @return None
//  */
// void set_bd_control(BD_PTR bd, uint32_t control);

// /**
//  * Set the status reg in the buffer descriptor
//  * @param bd Pointer to the buffer descriptor
//  * @param status Status value
//  * @return None
//  */
// void set_bd_status(BD_PTR bd, uint32_t status);

/**
 * Program this bd's buffer length field. Maximum size: 26 bits. Same for both S2MM and MM2s
 * @param bd Pointer to the buffer descriptor
 * @param len Length of the transfer in bytes
 * @return None
 */
void set_buffer_length(BD_PTR bd, uint32_t len);

/**
 * Set the SOF bit in the control reg of the buffer descriptor
 * Set this bit if it is the first bd in a ring
 * @param bd Pointer to the buffer descriptor
 * @param val Value (either 0 or 1)
 * @return None
 */
void set_sof_bit(BD_PTR bd, uint32_t val);

/**
 * Set the EOF bit in the control reg of the buffer descriptor
 * Set this bit if it is the last bd in a ring
 * @param bd Pointer to the buffer descriptor
 * @param val Value (either 0 or 1)
 * @return None
 */
void set_eof_bit(BD_PTR bd, uint32_t val);

/**
 * Clear the cmplt bit in the status reg in the buffer descriptor
 * @param bd Pointer to the buffer descriptor
 * @return None
 */
void clear_cmplt_bit(BD_PTR bd);

/**
 * Get the cmplt bit from the status reg in the buffer descriptor
 * @param bd Pointer to the buffer descriptor
 * @return Value of the cmplt bit
 */
uint32_t get_bd_cmplt_bit(BD_PTR bd);

/**
 * Return the number of bytes transferred by the given buffer descriptor
 * @param bd Pointer to the buffer descriptor
 * @return Number of bytes transferred
 */
uint32_t get_transferred_bytes(BD_PTR bd);

/**
 * Return the physical address of the buffer descriptor at the given index of the MM2S Ring
 * @param index Index of the buffer descriptor
 * @return Physical address of the buffer descriptor
 */
uint32_t mm2s_bd_idx_to_addr(uint32_t index);

/**
 * Return the physical address of the buffer descriptor at the given index of the S2MM Ring
 * @param index Index of the buffer descriptor
 * @return Physical address of the buffer descriptor
 */
uint32_t s2mm_bd_idx_to_addr(uint32_t index);


#endif // DMA_SG_BD_H_