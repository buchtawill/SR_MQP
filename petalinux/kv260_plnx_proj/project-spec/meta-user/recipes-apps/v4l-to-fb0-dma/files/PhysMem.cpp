#include <stdint.h>
#include <stdlib.h>
#include <string.h> //memcpy
#include "PhysMem.h"

PhysMem::PhysMem(volatile void* addr, size_t size, uint32_t id, uint32_t base_addr, bool unique_mmap){
    this->mem_id = id;
    this->mem_ptr = addr;
    this->num_bytes = size;
    this->base_address = base_addr;
    this->self_mmapped = unique_mmap;
}

PhysMem::~PhysMem(){
    // Memory freeing will be handled by PhysMman class
}

int PhysMem::write_from(void *src, size_t num_bytes){
    memcpy((void*)this->mem_ptr, src, num_bytes);
}

void PhysMem::read_into(void *dst, size_t num_bytes){
    memcpy(dst, (void*)this->mem_ptr, num_bytes);
}

void PhysMem::write_word(uint32_t byte_offset, uint32_t data){
    uint32_t address = byte_offset & (~0x00000003);
    ((uint32_t*)mem_ptr)[address] = data;
}

uint32_t PhysMem::read_word(uint32_t byte_offset){
    uint32_t address = byte_offset & (~0x00000003);
    return ((uint32_t*)mem_ptr)[address];
}

uint8_t PhysMem::read_byte(uint32_t byte_offset){
    return ((uint8_t*)mem_ptr)[byte_offset];
}

void PhysMem::write_word(uint8_t data, uint32_t byte_offset){
    ((uint8_t*)mem_ptr)[byte_offset] = data;
}

bool PhysMem::is_null(){
    if(this->mem_ptr == nullptr || this->base_address == NULL){
        return true;
    }

    return false;
}
