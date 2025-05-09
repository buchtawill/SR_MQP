
#include "phys-mman.h"
#include "PhysMem.h"
#include <stdlib.h>
#include <string.h> //memcpy
#include <sys/mman.h>

#include <errno.h>
#include <stdio.h>

#ifdef PLATFORM_HAS_RSVD_MEM
int PhysMman::init(int dev_mem_fd){

    if(!initialized){
        this->dev_mem_fd = dev_mem_fd;

        // uint32_t mem_size = PHYS_MMAN_NUM_CHUNKS * PHYS_MMAN_CHUNK_SIZE;

        // mmap the entire reserved memory
        volatile void* mem_ptr = (volatile void*)mmap(
            NULL,                   // Let the kernel decide the virtual address
            PMM_RSVD_MEM_SIZE,      // Size of memory to map 
            PROT_READ | PROT_WRITE, // Permissions: read and write
            MAP_SHARED,             // Changes are shared with other mappings
            this->dev_mem_fd,       // File descriptor for /dev/mem
            PMM_RSVD_MEM_BASE       // Physical address of the reserved memory
        );
        if(mem_ptr == MAP_FAILED){
            printf("ERROR [PhysMman::init()] Failed to map memory: %s\n", strerror(errno));
            return -1;
        }

        this->mem_base_ptr = mem_ptr;

        initialized = true;

        return 0;
    }

    printf("ERROR [PhysMman::init()] PhysMman is already initialized\n");
    return -1;
}

int PhysMman::self_test(){
    // Still need to implement
    return 0;
}
#else
int PhysMman::init(){

    if(!initialized){

        uint32_t mem_size = PHYS_MMAN_NUM_CHUNKS * PHYS_MMAN_CHUNK_SIZE;

        // For testing purposes
        volatile void* mem_ptr = (volatile void*)malloc(mem_size);

        this->mem_base_ptr = mem_ptr;
        initialized = true;

        return 0;
    }
    printf("ERROR [PhysMman::init()] PhysMman is already initialized\n");
    return -1;
}

int PhysMman::self_test(){
    // For test, don't need to pass in /dev/mem fd
    PMM.init(69);

    PMM.print_mem_blocks();
    PhysMem *block1 = PMM.alloc(1024);
    PhysMem *block2 = PMM.alloc(1000);
    PhysMem *block3 = PMM.alloc(10000);
    PhysMem *block4 = PMM.alloc(100*1024);

    block1->write_byte(4, 0xEF);
    block1->write_byte(5, 0xBE);
    block1->write_byte(6, 0x0D);
    block1->write_byte(7, 0xF0);

    uint32_t data;
    block1->read_word(4, &data);
    printf("INFO [main()] Block 1 word: 0x%08X\n", data);
    
    // std::cout<<"Block 1 size: "<< block_1->size() << std::endl;

    PMM.print_mem_blocks();
    PMM.free(block3);

    PMM.print_mem_blocks();

    PMM.free(block1);
    PMM.free(block2);
    PMM.free(block4);
}
#endif


PhysMem* PhysMman::alloc(size_t num_bytes, bool clear){

    if(!initialized){
        printf("ERROR [PhysMman::alloc()] PhysMman not initialized\n");
        return nullptr;
    }

    // Allocate to the next available block, don't care exactly where 
    // as long as it is a physically contiguous block

    // Calculate the number of chunks needed for num_bytes
    // Find the next available block that has the required number of chunks
    // from free_mem_blocks
    uint32_t num_chunks = (num_bytes / PHYS_MMAN_CHUNK_SIZE);
    if(num_bytes % PHYS_MMAN_CHUNK_SIZE != 0){
        num_chunks++;
    }
    size_t actual_allocated = num_chunks * PHYS_MMAN_CHUNK_SIZE;

    // Find the next available block that has the required number of chunks
    uint32_t first_chunk = 0;
    bool found = false;
    for(uint32_t i = 0; i < free_mem_blocks.size(); i++){
        
        // Reduce the number of free chunks from that entry
        if(free_mem_blocks[i].num_chunks >= num_chunks){
            found = true;
            first_chunk = free_mem_blocks[i].start_chunk;

            free_mem_blocks[i].start_chunk += num_chunks;
            free_mem_blocks[i].num_chunks  -= num_chunks;

            // Erase that entry if it's fully used
            if(free_mem_blocks[i].num_chunks == 0) 
                free_mem_blocks.erase(free_mem_blocks.begin() + i);

            break;
        }
    }
    if(!found){
        printf("ERROR [PhysMman::alloc()] Not enough memory available\n");
        return nullptr;
    }


    uint32_t base_addr = PMM_RSVD_MEM_BASE + (PHYS_MMAN_CHUNK_SIZE * first_chunk);

    // Brain f*ck line
    // Cast the mem base pointer to uint8_t, add the offset, then cast back to volatile void*
    // base_ptr is already mapped into the process
    volatile void* base_ptr = (volatile void*)((uint8_t*)(this->mem_base_ptr) + (PHYS_MMAN_CHUNK_SIZE * first_chunk));

    if(clear){
        memset((void*)base_ptr, 0, actual_allocated);
    }

    // Create an ID for the physical memory object
    uint32_t id = this->next_id;
    this->next_id++;

    PhysMem *block_ptr = new PhysMem(base_ptr, actual_allocated, id, base_addr, false);
    physmem_list.push_back(block_ptr);

    // Add the entry to the used blocks list
    PhysBlock pb_and_j;
    pb_and_j.start_chunk  = first_chunk;
    pb_and_j.num_chunks   = num_chunks;
    pb_and_j.physblock_id = id;
    used_mem_blocks.push_back(pb_and_j);

    return block_ptr;
}

PhysMem* PhysMman::alloc(uint32_t base_addr, size_t num_bytes){
    if(!initialized){
        printf("ERROR [PhysMman::alloc()] PhysMman not initialized\n");
        return nullptr;
    }

    // Allocate to a specific physical address (i.e., map a buffer)
    if(base_addr >= PMM_RSVD_MEM_BASE && 
        base_addr < PMM_RSVD_MEM_BASE + PMM_RSVD_MEM_SIZE)
    {
        printf("ERROR [PhysMman::alloc()] Cannot allocate to reserved memory. Use alloc(n_bytes) instead\n");
        return nullptr;
    }

    // mmap the base address and num bytes (rounded up to PHYS_MMAN_CHUNK_SIZE)
    uint32_t num_chunks = (num_bytes / PHYS_MMAN_CHUNK_SIZE);
    if(num_bytes % PHYS_MMAN_CHUNK_SIZE != 0){
        num_chunks++;
    }
    size_t actual_allocated = num_chunks * PHYS_MMAN_CHUNK_SIZE;

    printf("INFO [PhysMman::alloc()] Allocating %lu bytes to physical address 0x%08X\n", actual_allocated, base_addr);

    // Declare as volatile for no compiler optimizations --> immediate HW R/W
    volatile void* mem_ptr = (volatile void*)mmap(
        NULL,                   // Let the kernel decide the virtual address
        actual_allocated,       // Size of memory to map 
        PROT_READ | PROT_WRITE, // Permissions: read and write
        MAP_SHARED,             // Changes are shared with other mappings
        this->dev_mem_fd,       // File descriptor for /dev/mem
        base_addr               // Physical address of the reserved memory
    );
    if(mem_ptr == MAP_FAILED){
        printf("ERROR [PhysMman::alloc()] Failed to map memory: %s\n", strerror(errno));
        return nullptr;
    }

    // Create an ID for the physical memory object
    uint32_t id = this->next_id;
    this->next_id++;
    PhysMem *block_ptr = new PhysMem(mem_ptr, actual_allocated, id, base_addr, true);

    physmem_list.push_back(block_ptr);

    return block_ptr;
}

PhysMman::~PhysMman(){
    printf("INFO [PhysMman::~PhysMman()] Freeing all memory blocks\n");
    for(uint32_t i = 0; i < physmem_list.size(); i++){
        free(physmem_list[i]);
    }

    #ifdef PLATFORM_HAS_RSVD_MEM
    munmap((void*)(this->mem_base_ptr), PMM_RSVD_MEM_SIZE);
    #else
    free(this->mem_base_ptr);
    #endif
}

int PhysMman::free(PhysMem* mem){

    if(mem == nullptr){
        printf("ERROR [PhysMman::free()] Mem ptr is nullptr\n");
        return -1;
    }

    // Remove the mem from the list of mappings
    bool found = false;
    for(uint32_t i = 0; i < physmem_list.size(); i++){
        // printf(" List id: %d. mem id: %d\n", physmem_list[i]->get_id(), mem->get_id());
        if(physmem_list[i]->get_id() == mem->get_id()){
            physmem_list.erase(physmem_list.begin() + i);
            found = true;
            break;
        }
    }

    if(!found){
        printf("ERROR [PhysMman::free()] Memory block not found\n");
        return -1;
    }

    // If it has its own mmap, munmap it. Don't care about adding back to free block list
    if(mem->own_mmap() && !mem->is_null()){
        munmap((void*)mem->get_mem_ptr(), mem->size());
        delete mem;
        return 0;
    }

    // Add it back to the free chunks
    uint32_t num_freed_chunks = mem->size() / PHYS_MMAN_CHUNK_SIZE;
    uint32_t base_address = mem->get_phys_address();
    uint32_t start_chunk = (base_address - PMM_RSVD_MEM_BASE) / PHYS_MMAN_CHUNK_SIZE;

    // Check if the freed block is contiguous with any existing free blocks, and can be merged
    bool merged = false;
    for(uint32_t i = 0; i < free_mem_blocks.size(); i++){
        if(free_mem_blocks[i].start_chunk + free_mem_blocks[i].num_chunks == start_chunk){
            free_mem_blocks[i].num_chunks += num_freed_chunks;
            merged = true;
            break;
        }
        else if(free_mem_blocks[i].start_chunk - num_freed_chunks == start_chunk){
            free_mem_blocks[i].start_chunk -= num_freed_chunks;
            free_mem_blocks[i].num_chunks += num_freed_chunks;
            merged = true;
            break;
        }
    }

    // Otherwise, add it as a new free block (keeping the order / sortedness)
    if(!merged){
        PhysBlock pb;
        pb.start_chunk = start_chunk;
        pb.num_chunks = num_freed_chunks;
        for(uint32_t i = 0; i < free_mem_blocks.size(); i++){
            if(free_mem_blocks[i].start_chunk > start_chunk){
                free_mem_blocks.insert(free_mem_blocks.begin() + i, pb);
                break;
            }
        }
    }

    // Remove the used block entry with the same id
    for(uint32_t i = 0; i < used_mem_blocks.size(); i++){
        if(used_mem_blocks[i].physblock_id == mem->get_id()){
            used_mem_blocks.erase(used_mem_blocks.begin() + i);
            break;
        }
    }

    delete mem;
    return 0;
}

void PhysMman::print_mem_blocks(){
    printf("INFO [PhysMman::print_mem_blocks()] ~~~~~~~~~~~~~~~~~~~~\n");
    printf("  Free memory blocks \n");
    for(uint32_t i = 0; i < free_mem_blocks.size(); i++){
        printf("   - Start chunk: %5d, Num chunks: %5d\n", free_mem_blocks[i].start_chunk, free_mem_blocks[i].num_chunks);
    }

    printf("  Used memory blocks \n");
    for(uint32_t i = 0; i < used_mem_blocks.size(); i++){
        printf("   - Start chunk: %5d, Num chunks: %5d\n", used_mem_blocks[i].start_chunk, used_mem_blocks[i].num_chunks);
    }

    printf("\n");
}
