#ifndef PHYSMEM_H_
#define PHYSMEM_H_

#include <stdint.h>
#include <stdlib.h>

class PhysMem{
private:

    volatile void* mem_ptr;     // To be used by userspace application
    size_t num_bytes;           // Size of the allocated memory (multiple of 1kB)
    uint32_t mem_id;            // id to be used by PhysMman
    uint32_t base_address = 0;  // Physical address of memory
    bool self_mmapped = false;  // True if allocated from kernel reserved memory. False if mapped to HW

public:

    /**
     * Get the ID of the memory
     * @return ID of the memory
     */
    uint32_t get_id(){
        return mem_id;
    }

    /**
     * Get the physical address of the memory
     * @return Physical address of the memory
     */
    uint32_t get_phys_address(){
        return base_address;
    }

    /**
     * Get the pointer to the memory region to be used by userspace
     * @return Pointer to the memory region
     */
    volatile void* get_mem_ptr(){
        return mem_ptr;
    }

    /**
     * Return true if the memory has its own mmap call
     * @return true if the memory is self mmapped
     */
    bool own_mmap() { return this->self_mmapped; }

    /**
     * Return the size of the memory in bytes
     */
    size_t size(){
        return num_bytes;
    }

    PhysMem(volatile void* addr, size_t size, uint32_t id, uint32_t base_addr, bool unique_mmap);

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

#endif // PHYSMEM_H_
