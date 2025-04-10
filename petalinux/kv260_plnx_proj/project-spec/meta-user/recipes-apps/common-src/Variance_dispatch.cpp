#include "Variance_dispatch.hpp"
#include "stdint-gcc.h"
#include "phys-mman.h"

VarianceDispatcher::VarianceDispatcher(uint32_t base_addr, uint32_t dev_mem_fd){
    this->base_address_phys = base_addr;
    this->dev_mem_fd = dev_mem_fd;
}

VarianceDispatcher::~VarianceDispatcher(){
    PMM.free(this->mem_block);
}

int VarianceDispatcher::init(){

    // Initialize the PhysMem block and confirm read and write ability to control
    this->mem_block = PMM.alloc(this->base_address_phys, 0x10000);
    if(this->mem_block == nullptr){
        puts("ERROR [VarianceDispatcher::init()] Failed to allocate physical memory block\n");
        return -1;
    }

    return 0;
}

void VarianceDispatcher::start(){
    uint32_t data;
    this->mem_block->read_word(XPROCESS_TILE_CONTROL_ADDR_AP_CTRL, &data);
    data |= (uint32_t)0x1;
    this->mem_block->write_word(XPROCESS_TILE_CONTROL_ADDR_AP_CTRL, data);
}

uint32_t VarianceDispatcher::is_done(){
    uint32_t control;
    this->mem_block->read_word(XPROCESS_TILE_CONTROL_ADDR_AP_CTRL, &control);
    return (control >> 1) & 0x1;
}

uint32_t VarianceDispatcher::is_idle(){
    uint32_t control;
    this->mem_block->read_word(XPROCESS_TILE_CONTROL_ADDR_AP_CTRL, &control);
    return (control >> 2) & 0x1;
}

uint32_t VarianceDispatcher::is_ready(){
    uint32_t control;
    this->mem_block->read_word(XPROCESS_TILE_CONTROL_ADDR_AP_CTRL, &control);
    return !(control & 0x1);
}

void VarianceDispatcher::enable_auto_restart(){
    uint32_t data = 0x80;
    this->mem_block->write_word(XPROCESS_TILE_CONTROL_ADDR_AP_CTRL, data);
}

void VarianceDispatcher::disable_auto_restart(){
    uint32_t data = 0x0;
    this->mem_block->write_word(XPROCESS_TILE_CONTROL_ADDR_AP_CTRL, data);
}

void VarianceDispatcher::set_threshold(uint32_t value){
    this->mem_block->write_word(XPROCESS_TILE_CONTROL_ADDR_THRESHOLD_DATA, value);
}

uint32_t VarianceDispatcher::get_threshold(){
    uint32_t data;
    uint32_t result = this->mem_block->read_word(XPROCESS_TILE_CONTROL_ADDR_THRESHOLD_DATA, &data);
}

void VarianceDispatcher::set_override(uint32_t mode){
    this->mem_block->write_word(XPROCESS_TILE_CONTROL_ADDR_OVERRIDE_MODE_DATA, mode);
}

uint32_t VarianceDispatcher::get_override(){
    uint32_t data;
    uint32_t result = this->mem_block->read_word(XPROCESS_TILE_CONTROL_ADDR_OVERRIDE_MODE_DATA, &data);
}
