/**
 * This class serves to provide memory management for physical memory. This class is a singleton.
 * Each memory block will be represented by a class, PhysMem. It will contain a volatile void pointer 
 * that can be used by the userspace application to r/w the physical memory, and a uint32_t representing
 * the physical address to be used by a DMA engine. All PhysMems are guarunteed to be contiguous.
 * 
 * This PhysMem objects are mainly interpreted to be used as buffers.
 * 
 * The PhysMman class will create and keep track of buffers. Buffers can be created in an arbitrary physical memory space
 * by specifying the size of the buffer only, or can be created at a specific location by giving a base address and size.
 * PhysMem objects are destroyed and freed by the PhysMman class by 
 * Use cases: 
 *      - map frame buffer to user space --> specify base address and size
 *      - create a buffer, don't care where
 * 
 * Example
 *      PhysMem *rgb565_buf = PhysMman.create(RGB565_BUF_SIZE_BYTES);
 *      PhysMem *fb0_buf    = PhysMman.create(fixed_fb_info.smem_start, fb_size_bytes);
 */

#ifndef PHYS_MMAN_H
#define PHYS_MMAN_H

#include <stdint.h>
#include <stdlib.h>         // For size_t
#include <vector>

#define KERNEL_RSVD_MEM_BASE		0x78000000
#define KERNEL_RSVD_MEM_SIZE        0x02000000

/**
 * TODO: Decide if this class should use boundary tag or bitfields to keep track of memory
 */
class PhysMman {
private:

    std::vector<PhysMem*> used_mem;
};

class PhysMem{
private:

    volatile void* mem_ptr;     // To be used by userspace application
    size_t num_bytes;           // Size of the allocated memory (multiple of 1kB)
    uint32_t mem_id;            // id to be used by PhysMman
    uint32_t base_address;      // Physical address of memory

public:

    uint32_t get_phys_address(){
        return base_address;
    }

    volatile void* get_mem_ptr(){
        return mem_ptr;
    }

    /**
     * Return the size of the memory in bytes
     */
    size_t size(){
        return num_bytes;
    }

    PhysMem(volatile void* addr, size_t size, uint32_t id, uint32_t base_addr);

    /**
     * Memory freeing will be handled by PhysMman class
     */
    ~PhysMem();

    /**
     * Copy num_bytes bytes from src into the base of this memory
     * @param src Source address of data
     * @param num_bytes Number of bytes to copy
     * @return number of bytes copied on success, -1 on error
     */
    int write_from(void *src, size_t num_bytes);

    /**
     * Copy num_bytes bytes from this memory into dst
     * @param dst Destination of memory
     * @param num_bytes Number of bytes to copy/read
     */
    void read_into(void *dst, size_t num_bytes);

    /**
     * Write a word (data) to the specified address offset (byte_offset). Buyer beware - write to aligned address.
     * @param byte_offset Byte address in the memory
     * @param data Word to write
     */
    void write_word(uint32_t byte_offset, uint32_t data);
    
    /**
     * Read a word from the relevant 
     */
    uint32_t read_word(uint32_t byte_offset);

    /**
     * Read the byte from the address at byte_offset
     * @param byte_offset Memory address offset
     */
    uint8_t read_byte(uint32_t byte_offset);

    /**
     * Write data to byte_offset. Buyer beware - byte_offset must be aligned to 4 byte boundary
     */
    void write_word(uint8_t data, uint32_t byte_offset);
    
    /**
     * Return true if the memory points to Null
     * @return true if the memory is not allocated
     */
    bool is_null();
};


#endif // PHYS_MMAN_H