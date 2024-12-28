/**
 * This class serves to provide memory management for physical memory. This class is a singleton.
 * Each memory block will be represented by a container class, PhysMem. It will contain a volatile void pointer 
 * that can be used by the userspace application to r/w the physical memory, and a uint32_t representing
 * the physical address (which can be used by a DMA engine). All PhysMems are guarunteed to be have contiguous memory.
 * 
 * This PhysMem objects are mainly interpreted to be used as buffers or mappings to hardware registers.
 * 
 * The PhysMman class will create and keep track of buffers. Buffers can be created in an arbitrary physical memory space
 * by specifying the size of the buffer only, or can be created at a specific location by giving a base address and size.
 * PhysMem objects are destroyed and freed by the PhysMman class.
 * Use cases: 
 *      - map frame buffer to user space --> specify base address and size
 *      - create a buffer, don't care where --> specify size only
 * 
 * Example
 *      PhysMem *rgb565_buf = PhysMman.create(RGB565_BUF_SIZE_BYTES);
 *      PhysMem *fb0_buf    = PhysMman.create(fixed_fb_info.smem_start, fb_size_bytes);
 * 
 * All memory will be allocated in chunks of CHUNK_SIZE, defined in this header file
 * 
 * Author: Will Buchta
 * Date:   23 December 2024 (Merry Christmas)
 */

#ifndef PHYS_MMAN_H
#define PHYS_MMAN_H

#include <stdint.h>
#include <stdlib.h> // For size_t
#include <vector>

#include "PhysMem.h"

#define PMM_RSVD_MEM_BASE       0x78000000
#define PMM_RSVD_MEM_SIZE       0x08000000

// All PhysMem blocks will be aligned to CHUNK_SIZE byte boundary
#define PHYS_MMAN_CHUNK_SIZE    256
#define PHYS_MMAN_NUM_CHUNKS    (PMM_RSVD_MEM_SIZE / PHYS_MMAN_CHUNK_SIZE)

#define PLATFORM_HAS_RSVD_MEM 1 // For running on kv260
// #define PLATFORM_NO_RSVD_MEM // For running on other 

// Singleton class alias
#define PMM PhysMman::get_instance()

/**
 * This class will allocate memory using the bitmask method, using an array of booleans to keep track of allocated memory
 * The memory will be allocated in chunks of CHUNK_SIZE.
 * A "chunk" is a CHUNK_SIZE area of memory
 * A "block" is a number of contiguous chunks
 */
class PhysMman {
private:

    typedef struct{
        uint32_t start_chunk;
        uint32_t num_chunks;
        uint32_t physblock_id;
    } PhysBlock;

    // Keep track of what chunks are available
    std::vector<PhysBlock> free_mem_blocks;

    // Keep track of what chunks are used
    std::vector<PhysBlock> used_mem_blocks;

    // Maintain a list of allocated PhysMem objects
    std::vector<PhysMem*> physmem_list;

    volatile void *mem_base_ptr;    // Pointer to the reserved memory
    int next_id = 0;                // Next id to be assigned to a memory block

    bool initialized = false; // Whether or not this class has been init'd yet

    #ifdef PLATFORM_HAS_RSVD_MEM
    int dev_mem_fd;                 // File descriptor for /dev/mem
    #endif

    // Private constructor for singleton
    PhysMman(){
        free_mem_blocks.clear();
        used_mem_blocks.clear();

        physmem_list.clear();

        PhysBlock init;
        init.num_chunks  = PHYS_MMAN_NUM_CHUNKS;
        init.start_chunk = 0;
        free_mem_blocks.push_back(init);
    }
    PhysMman(PhysMman const&);          // Don't allow copy
    void operator=(PhysMman const&);    // Don't allow assignment

public:

    /**
     * Free / unmap all the memory blocks that were allocated by this class
     */
    ~PhysMman();

    static PhysMman& get_instance(){
        static PhysMman instance;
        return instance;
    }

    #ifdef PLATFORM_HAS_RSVD_MEM
    /**
     * Initialize the memory manager with the file descriptor for /dev/mem and
     * mmap the region
     * @param dev_mem_fd File descriptor for /dev/mem
     * @return 0 on success, -1 on error
     */
    int init(int dev_mem_fd);
    #else
    /**
     * Initialize the memory manager by using a simple malloc call
     * @return 0 on success, -1 on error
     */
    int init(int dev_mem_fd);
    #endif

    /**
     * Create a new memory block of size num_bytes. The memory will be allocated from the kernel reserved memory
     * @param num_bytes Size of the memory block in bytes
     * @return Pointer to the allocated memory block, or nullptr on error
     */
    PhysMem* alloc(size_t num_bytes);

    /**
     * Create a PhysMem instance that points to the specified physical address and is of size num_bytes.
     * Will round up 4kB of space
     * @param base_addr Base address of the memory block
     * @param num_bytes Size of the memory block in bytes
     * @return Pointer to the allocated memory block, or nullptr on error
     */
    PhysMem* alloc(uint32_t base_addr, size_t num_bytes);

    /**
     * Free the memory block pointed to by the PhysMem object
     * @param mem Pointer to the memory block to be freed
     * @return 0 on success, -1 on error
     */
    int free(PhysMem* mem);

    /**
     * Print the memory blocks that are free and used
     * @return None
     */
    void print_mem_blocks();

    /**
     * Perform a basic self test on the PhysMman class.
     * @return 0 on success, -1 on error
     */
    int self_test();
};

#endif // PHYS_MMAN_H
