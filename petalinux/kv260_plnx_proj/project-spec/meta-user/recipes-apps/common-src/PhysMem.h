#ifndef PHYSMEM_H_
#define PHYSMEM_H_

#include <stdint.h>
#include <stdlib.h>

#include <string.h> //memcpy
#include <stdio.h>

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
    inline uint32_t get_id(){
        return mem_id;
    }

    /**
     * Set the pointer to the memory region
     * @param ptr Pointer to the memory region
     */
    void set_mem_ptr(volatile void* ptr){
        this->mem_ptr = ptr;
    }

    /**
     * Get the physical address of the memory
     * @return Physical address of the memory
     */
    inline uint32_t get_phys_address(){
        return base_address;
    }

    /**
     * Get the pointer to the memory region to be used by userspace
     * @return Pointer to the memory region
     */
    inline volatile void* get_mem_ptr(){
        return mem_ptr;
    }

    /**
     * Return true if the memory has its own mmap call
     * @return true if the memory is self mmapped
     */
    inline bool own_mmap() { return this->self_mmapped; }

    /**
     * Return the size of the memory in bytes
     */
    inline size_t size(){
        return num_bytes;
    }

    /**
     * Constructor for PhysMem class
     * @param addr Pointer to the memory region
     * @param size Size of the memory region in bytes
     * @param id ID of the memory region (used by PhysMman)
     * @param base_addr Physical address of the memory region
     * @param unique_mmap True if the memory is allocated using mmap
     */
    PhysMem(volatile void* addr, size_t size, uint32_t id, uint32_t base_addr, bool unique_mmap){
        this->mem_id = id;
        this->mem_ptr = addr;
        this->num_bytes = size;
        this->base_address = base_addr;
        this->self_mmapped = unique_mmap;
    }

    /**
     * Memory freeing will be handled by PhysMman class
     */
    ~PhysMem(){
        // Memory freeing is handled by PhysMman class
    }

    /**
     * Copy num_bytes bytes from src into the base of this memory
     * @param src Source address of data
     * @param num_bytes Number of bytes to copy
     * @return number of bytes copied on success, -1 on error
     */
    inline int write_from(void *src, size_t num_bytes){

        if(num_bytes > this->num_bytes){
            printf("ERROR [PhysMem::write_from()] Requested write exceeds memory size\n");
            return -1;
        }

        memcpy((void*)this->mem_ptr, src, num_bytes);
        return num_bytes;
    }

    /**
     * Copy num_bytes bytes from this memory into dst
     * @param dst Destination of memory
     * @param num_bytes Number of bytes to copy/read
     * @return number of bytes copied on success, -1 on error
     */
    inline int read_into(void *dst, size_t num_bytes){

        if(num_bytes > this->num_bytes){
            printf("ERROR [PhysMem::read_into()] Requested read exceeds memory size\n");
            return -1;
        }

        memcpy(dst, (void*)this->mem_ptr, num_bytes);
        return num_bytes;
    }
    
    /**
     * Write a word (data) to the specified address offset (byte_offset). Buyer beware - write to aligned address.
     * @param byte_offset Byte address in the memory
     * @param data Word to write
     * @return 0 on success, -1 on error
     */
    inline int write_word(uint32_t byte_offset, uint32_t data){

        if(byte_offset > this->num_bytes-4){
            printf("ERROR [PhysMem::write_word()] Byte offset exceeds memory size\n");
            return -1;
        }

        uint32_t address = byte_offset & (~0x00000003);
        ((uint32_t*)mem_ptr)[address >> 2] = data;

        return 0;
    }
    
    /**
     * Read a word from the specified byte address
     * @param byte_offset Byte address in the memory
     * @param data Pointer to store the word
     */
    inline int read_word(uint32_t byte_offset, uint32_t *data){

        if(byte_offset > this->num_bytes-4){
            printf("ERROR [PhysMem::read_word()] Byte offset exceeds memory size\n");
            return -1;
        }

        uint32_t address = byte_offset & (~0x00000003);
        *data = ((uint32_t*)mem_ptr)[address >> 2];

        return 0;
    }

    /**
     * Read the byte from the address at byte_offset
     * @param byte_offset Memory address offset
     * @param data Byte to read
     */
    inline int read_byte(uint32_t byte_offset, uint8_t *data){

        if(byte_offset > this->num_bytes-1){
            printf("ERROR [PhysMem::read_byte()] Byte offset exceeds memory size\n");
            return -1;
        }

        *data = ((uint8_t*)mem_ptr)[byte_offset];
        return 0;
    }

    /**
     * Write data to byte_offset. Buyer beware - byte_offset must be aligned to 4 byte boundary
     * @param byte_offset Byte address in the memory
     * @param data Byte to write
     * @return 0 on success, -1 on error
     */
    inline int write_byte(uint32_t byte_offset, uint8_t data){

        if(byte_offset > this->num_bytes-1){
            printf("ERROR [PhysMem::write_byte()] Byte offset exceeds memory size\n");
            return -1;
        }

        ((uint8_t*)mem_ptr)[byte_offset] = data;

        return 0;
    }

    /**
     * Return true if the memory points to Null
     * @return true if the memory is not allocated
     */
    inline bool is_null(){
        if(this->mem_ptr == nullptr || this->base_address == 0){
            return true;
        }

        return false;
    }
};

#endif // PHYSMEM_H_
