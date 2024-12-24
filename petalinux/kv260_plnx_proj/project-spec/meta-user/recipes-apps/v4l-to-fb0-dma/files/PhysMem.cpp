#include <stdint.h>
#include <stdlib.h>
#include <string.h> //memcpy
#include "PhysMem.h"

#include <stdio.h>

PhysMem::PhysMem(volatile void* addr, size_t size, uint32_t id, uint32_t base_addr, bool unique_mmap){
    this->mem_id = id;
    this->mem_ptr = addr;
    this->num_bytes = size;
    this->base_address = base_addr;
    this->self_mmapped = unique_mmap;
}

PhysMem::~PhysMem(){
    // Memory freeing is handled by PhysMman class
}

int PhysMem::write_from(void *src, size_t num_bytes){

    if(num_bytes > this->num_bytes){
        printf("ERROR [PhysMem::write_from()] Requested write exceeds memory size\n");
        return -1;
    }

    memcpy((void*)this->mem_ptr, src, num_bytes);
    return num_bytes;
}

int PhysMem::read_into(void *dst, size_t num_bytes){

    if(num_bytes > this->num_bytes){
        printf("ERROR [PhysMem::read_into()] Requested read exceeds memory size\n");
        return -1;
    }

    memcpy(dst, (void*)this->mem_ptr, num_bytes);
    return num_bytes;
}

int PhysMem::write_word(uint32_t byte_offset, uint32_t data){

    if(byte_offset > this->num_bytes-4){
        printf("ERROR [PhysMem::write_word()] Byte offset exceeds memory size\n");
        return -1;
    }

    uint32_t address = byte_offset & (~0x00000003);
    ((uint32_t*)mem_ptr)[address >> 2] = data;

    return 0;
}

int PhysMem::read_word(uint32_t byte_offset, uint32_t *data){

    if(byte_offset > this->num_bytes-4){
        printf("ERROR [PhysMem::read_word()] Byte offset exceeds memory size\n");
        return -1;
    }

    uint32_t address = byte_offset & (~0x00000003);
    *data = ((uint32_t*)mem_ptr)[address >> 2];

    return 0;
}

int PhysMem::read_byte(uint32_t byte_offset, uint8_t *data){

    if(byte_offset > this->num_bytes-1){
        printf("ERROR [PhysMem::read_byte()] Byte offset exceeds memory size\n");
        return -1;
    }

    *data = ((uint8_t*)mem_ptr)[byte_offset];
    return 0;
}

int PhysMem::write_byte(uint32_t byte_offset, uint8_t data){

    if(byte_offset > this->num_bytes-1){
        printf("ERROR [PhysMem::write_byte()] Byte offset exceeds memory size\n");
        return -1;
    }

    ((uint8_t*)mem_ptr)[byte_offset] = data;

    return 0;
}

bool PhysMem::is_null(){
    if(this->mem_ptr == nullptr || this->base_address == 0){
        return true;
    }

    return false;
}
