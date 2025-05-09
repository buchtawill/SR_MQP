/*

This program will read frames from a video device, convert them from YUYV 4:2:2 to RGB888, interpolate them with our HW, then 
convert to RGB565 before displaying them on the framebuffer device.

This thing is going to be so ungodly slow, but it is just a prototype. The main flow is this:
 1. Capture a frame from v4l2
 2. Convert the frame to RGB888
 - Loop:
   3. Send a tile down to the PL, one line at a time
   4. Receive the tile and store it in a temporary buffer (can be in userspace)
   5. Convert the tile to RGB565
   6. Copy the tile to the frame buffer, one line at a time

Steps 2, 4, 5, and 6 will all be deleted eventually

Author: Will Buchta
Date Modified: 1/11/2025

*/

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>

#include <fcntl.h> // for open flags

#include <sys/ioctl.h>
// #include <linux/videodev2.h>
#include <linux/fb.h>
#include <sys/mman.h> // TEMPORARY solution for v4l2 buffer mapping

#include <signal.h> //for sigint
#include <time.h>   // for jiffies

#include "upscale-zcu102.h"
#include "phys-mman.h"
#include "PhysMem.h"

#include "axi-dma.h"
#include "dma-sg-bd.h"

#include "Variance_dispatch.hpp"

#include "argparse.hpp"

// Global variable - use with caution! This should only ever be set to true 
bool die_flag = false;

unsigned long get_jiffies(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long)(ts.tv_sec * sysconf(_SC_CLK_TCK) + ts.tv_nsec / (1e9 / sysconf(_SC_CLK_TCK)));
}

// Convert YUYV to RGB565
// Verified to work
void yuyv_to_rgb565(uint8_t *yuyv, uint16_t *rgb565, int width, int height) {
    for (int i = 0; i < width * height; i += 2) {
        int y0 = yuyv[i * 2];
        int u = yuyv[i * 2 + 1] - 128;
        int y1 = yuyv[i * 2 + 2];
        int v = yuyv[i * 2 + 3] - 128;

        int r0 = y0 + 1.403 * v;
        int g0 = y0 - 0.344 * u - 0.714 * v;
        int b0 = y0 + 1.770 * u;

        int r1 = y1 + 1.403 * v;
        int g1 = y1 - 0.344 * u - 0.714 * v;
        int b1 = y1 + 1.770 * u;

        r0 = r0 < 0 ? 0 : (r0 > 255 ? 255 : r0);
        g0 = g0 < 0 ? 0 : (g0 > 255 ? 255 : g0);
        b0 = b0 < 0 ? 0 : (b0 > 255 ? 255 : b0);

        r1 = r1 < 0 ? 0 : (r1 > 255 ? 255 : r1);
        g1 = g1 < 0 ? 0 : (g1 > 255 ? 255 : g1);
        b1 = b1 < 0 ? 0 : (b1 > 255 ? 255 : b1);

        rgb565[i] = ((r0 & 0xF8) << 8) | ((g0 & 0xFC) << 3) | (b0 >> 3);
        rgb565[i + 1] = ((r1 & 0xF8) << 8) | ((g1 & 0xFC) << 3) | (b1 >> 3);
    }
}

// Convert YUYV to RGB888
// Verified to work
void yuyv_to_rgb888(uint8_t *yuyv, uint8_t *rgb888, int width, int height) {
    for (int i = 0; i < width * height; i += 2) {
        int y0 = yuyv[i * 2];
        int u = yuyv[i * 2 + 1] - 128;
        int y1 = yuyv[i * 2 + 2];
        int v = yuyv[i * 2 + 3] - 128;

        int r0 = y0 + 1.403 * v;
        int g0 = y0 - 0.344 * u - 0.714 * v;
        int b0 = y0 + 1.770 * u;

        int r1 = y1 + 1.403 * v;
        int g1 = y1 - 0.344 * u - 0.714 * v;
        int b1 = y1 + 1.770 * u;

        r0 = r0 < 0 ? 0 : (r0 > 255 ? 255 : r0);
        g0 = g0 < 0 ? 0 : (g0 > 255 ? 255 : g0);
        b0 = b0 < 0 ? 0 : (b0 > 255 ? 255 : b0);

        r1 = r1 < 0 ? 0 : (r1 > 255 ? 255 : r1);
        g1 = g1 < 0 ? 0 : (g1 > 255 ? 255 : g1);
        b1 = b1 < 0 ? 0 : (b1 > 255 ? 255 : b1);

        rgb888[i * 3] = r0;
        rgb888[i * 3 + 1] = g0;
        rgb888[i * 3 + 2] = b0;
        rgb888[(i + 1) * 3] = r1;
        rgb888[(i + 1) * 3 + 1] = g1;
        rgb888[(i + 1) * 3 + 2] = b1;
    }
}

/**
 * Convert num_pixels pixels from RGB888 to RGB565, moving from rgb888 buffer to rgb565 buffer
 * @param rgb888 Pointer to the RGB888 pixels
 * @param rgb565 Pointer to the RGB565 pixels
 * @param num_pixels The number of pixels to move
*/
void rgb888_to_rgb565(uint8_t *rgb888, uint16_t *rgb565, uint32_t num_pixels) {

    // i is pixel index
    for (uint32_t i = 0; i < num_pixels; i++) {
        uint8_t r = rgb888[i * 3];
        uint8_t g = rgb888[i * 3 + 1];
        uint8_t b = rgb888[i * 3 + 2];

        rgb565[i] = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
    }
}

/**
 * Print a given error message and clean up resources before exiting
 * @param error_msg The error message to print
 * @param err_str The error string to print (optional, given by strerror(errno), set to nullptr if not using)
 * @param res Pointer to the resources struct
 * @return None
 */
void die_with_error(const char* error_msg, const char* err_str, Resources* res){
    die_flag = true;

    printf("%s", error_msg);

    if(err_str != nullptr){
        printf("%s\n", err_str);
    }
    else{
        printf("\n");
    }

    cleanup_resources(res);
    exit(-1);
}

void sigint_handler(int sig){

    if(sig == SIGINT){
        printf("INFO [upscale-zcu102] Received signal, cleaning up...\n");
        die_flag = true;
    }
}

/**
 * Open dev/mem
 * Open the video device
 * Open the framebuffer device
 * Set up buffers for the RGB888 frame and interpolated RGB888 frame
 * @param p_res Pointer to the resources struct
 * @param fb_device The framebuffer device
 * @return None
 */
void init_resources(Resources *p_res, const char* fb_dev){
    p_res->v4l2_fd         = -1;
    p_res->fb_dev_fd       = -1;
    p_res->dev_mem_fd      = -1;
    // p_res->vid_mem_block   = nullptr;
    p_res->vid_mem_ptr[0]  = nullptr;
    p_res->vid_mem_ptr[1]  = nullptr;
    p_res->fb_mem_block    = nullptr;
    p_res->input888_block  = nullptr;
    p_res->interp888_block = nullptr;

    // memset(&p_res->v4l2_fmt,             0, sizeof(p_res->v4l2_fmt));
    // memset(&p_res->v4l2_req,             0, sizeof(p_res->v4l2_req));
    memset(&p_res->fixed_fb_info,        0, sizeof(p_res->fixed_fb_info));
    // memset(&p_res->v4l2_frame_bufs[0],   0, sizeof(p_res->v4l2_frame_bufs[0]));
    // memset(&p_res->v4l2_frame_bufs[1],   0, sizeof(p_res->v4l2_frame_bufs[1]));
    memset(&p_res->configurable_fb_info, 0, sizeof(p_res->configurable_fb_info));

    p_res->fb_phys_base           = 0;
    p_res->fb_size_bytes          = 0;
    p_res->vid_mem_size_bytes     = 0;

    // Open /dev/mem
	printf("INFO [upscale-zcu102::init_resources()] Opening /dev/mem\n");
	p_res->dev_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
	if(p_res->dev_mem_fd < 0){
		die_with_error("ERROR [upscale-zcu102::init_resources()] Error opening /dev/mem: ", strerror(errno), p_res);
	}

    // Initialize the Physical Memory Manager
    printf("INFO [upscale-zcu102::init_resources()] Initializing Physical Memory Manager\n");
    int pmm_result = PMM.init(p_res->dev_mem_fd);
    if(pmm_result != 0){
        die_with_error("ERROR [upscale-zcu102::init_resources()] Failed to initialize Physical Memory Manager\n", nullptr, p_res);
    }

    // Open the video device
    // printf("INFO [upscale-zcu102::init_resources()] Opening video device %s\n", video_dev);
    // p_res->v4l2_fd = open(video_dev, O_RDWR);
    // if(p_res->v4l2_fd < 0){
    //     die_with_error("ERROR [upscale-zcu102::init_resources()] Error opening video device: ", strerror(errno), p_res);
    // }

    // Open the framebuffer device
    printf("INFO [upscale-zcu102::init_resources()] Opening framebuffer device %s\n", fb_dev);
    p_res->fb_dev_fd = open(fb_dev, O_RDWR);
    if(p_res->fb_dev_fd < 0){
        die_with_error("ERROR [upscale-zcu102::init_resources()] Error opening framebuffer device: ", strerror(errno), p_res);
    }

    // Create buffers for the RGB888 frame and interpolated RGB888 frame
    p_res->input888_block = PMM.alloc(INPUT_VIDEO_HEIGHT * INPUT_VIDEO_WIDTH * 3);
    if(p_res->input888_block == nullptr){
        die_with_error("ERROR [upscale-zcu102::init_resources()] Failed to allocate input888 block\n", nullptr, p_res);
    }
    printf("INFO [upscale-zcu102::init_resources()] Allocated input888 block to address 0x%08X\n", p_res->input888_block->get_phys_address());

    p_res->interp888_block = PMM.alloc(INPUT_VIDEO_HEIGHT * UPSCALE_FACTOR * INPUT_VIDEO_WIDTH * UPSCALE_FACTOR * 3);
    if(p_res->interp888_block == nullptr){
        die_with_error("ERROR [upscale-zcu102::init_resources()] Failed to allocate interp888 block\n", nullptr, p_res);
    }
    printf("INFO [upscale-zcu102::init_resources()] Allocated interp888 block to address 0x%08X\n", p_res->interp888_block->get_phys_address());
}

/**
 * Close and unmap all files and memory associated with the resources struct.
 * @param p_res Pointer to the resources struct
 * @return None
 */
void cleanup_resources(Resources *p_res){

    // if(p_res->v4l2_fd != -1){
    //     close(p_res->v4l2_fd);
    // }
    // if(p_res->fb_dev_fd != -1){
    //     close(p_res->fb_dev_fd);
    // }
    if(p_res->dev_mem_fd != -1){
        close(p_res->dev_mem_fd);
    }

    // if (p_res->vid_mem_ptr[0] != MAP_FAILED) {
    //     munmap(p_res->vid_mem_ptr, p_res->vid_mem_size_bytes);
    // }

    // if (p_res->vid_mem_ptr[1] != MAP_FAILED) {
    //     munmap(p_res->vid_mem_ptr, p_res->vid_mem_size_bytes);
    // }

    // ioctl(p_res->v4l2_fd, VIDIOC_STREAMOFF, &p_res->v4l2_frame_buf.type);

    // PMM.free(p_res->vid_mem_block);
    PMM.free(p_res->fb_mem_block);
    PMM.free(p_res->input888_block);
    PMM.free(p_res->interp888_block);
}

/**
 * Setup all things v4l2.
 * Open the video device
 * Set the video format
 * Request buffers from the video device
 * Map the video buffer
 * @param p_res Pointer to the resources struct
 * @return None
 */
/*
void setup_video(Resources *p_res){

    // Set video format
    p_res->v4l2_fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    p_res->v4l2_fmt.fmt.pix.width       = INPUT_VIDEO_WIDTH;
    p_res->v4l2_fmt.fmt.pix.height      = INPUT_VIDEO_HEIGHT;
    p_res->v4l2_fmt.fmt.pix.pixelformat = INPUT_VIDEO_PIXEL_FMT;
    p_res->v4l2_fmt.fmt.pix.field       = V4L2_FIELD_NONE;

    printf("INFO [upscale-zcu102::setup_video()] Setting video format\n");
    if (ioctl(p_res->v4l2_fd, VIDIOC_S_FMT, &p_res->v4l2_fmt) == -1) {
        die_with_error("ERROR [upscale-zcu102::setup_video()] Error setting video format: ", strerror(errno), p_res);
    }

    // Request buffers from the video device
    printf("INFO [upscale-zcu102::setup_video()] Requesting buffer from video device\n");
    p_res->v4l2_req.count = 2; // Use 1 buffer
    p_res->v4l2_req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    p_res->v4l2_req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(p_res->v4l2_fd, VIDIOC_REQBUFS, &p_res->v4l2_req) == -1) {
        die_with_error("ERROR [upscale-zcu102::setup_video()] Error requesting v4l2 buffer: ", strerror(errno), p_res);
    }

    // Map the video buffer s
    printf("INFO [upscale-zcu102::setup_video()] Mapping video buffer\n");
    p_res->v4l2_frame_bufs[0].type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    p_res->v4l2_frame_bufs[0].memory = V4L2_MEMORY_MMAP;
    p_res->v4l2_frame_bufs[0].index = 0;

    p_res->v4l2_frame_bufs[1].type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    p_res->v4l2_frame_bufs[1].memory = V4L2_MEMORY_MMAP;
    p_res->v4l2_frame_bufs[1].index = 1;

    if (ioctl(p_res->v4l2_fd, VIDIOC_QUERYBUF, &p_res->v4l2_frame_bufs[0]) == -1) {
        die_with_error("ERROR [upscale-zcu102::setup_video()] Error querying v4l2 buffer 0: ", strerror(errno), p_res);
    }

    if (ioctl(p_res->v4l2_fd, VIDIOC_QUERYBUF, &p_res->v4l2_frame_bufs[1]) == -1) {
        die_with_error("ERROR [upscale-zcu102::setup_video()] Error querying v4l2 buffer 1: ", strerror(errno), p_res);
    }

    p_res->vid_mem_size_bytes = p_res->v4l2_frame_bufs[0].length;

    // Create two buffers
    // p_res->vid_mem_blocks[0] = PMM.alloc(p_res->vid_mem_size_bytes);
    // p_res->vid_mem_blocks[1] = PMM.alloc(p_res->vid_mem_size_bytes);
    // if(p_res->vid_mem_blocks[0] == nullptr || p_res->vid_mem_blocks[1] == nullptr){
    //     die_with_error("ERROR [upscale-zcu102::setup_video()] Failed to map video buffer to PhysMem object\n", nullptr, p_res);
    // }

    // Map the video buffers
    p_res->vid_mem_ptr[0] = mmap(NULL, p_res->vid_mem_size_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, p_res->v4l2_fd, p_res->v4l2_frame_bufs[0].m.offset);
    p_res->vid_mem_ptr[1] = mmap(NULL, p_res->vid_mem_size_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, p_res->v4l2_fd, p_res->v4l2_frame_bufs[1].m.offset);
    printf("INFO [upscale-zcu102::setup_video()] Buffer 0 offset: 0x%x\n", p_res->v4l2_frame_bufs[0].m.offset);
    printf("INFO [upscale-zcu102::setup_video()] Buffer 1 offset: 0x%x\n", p_res->v4l2_frame_bufs[1].m.offset);
    
    if (p_res->vid_mem_ptr[0] == MAP_FAILED || p_res->vid_mem_ptr[1] == MAP_FAILED) {
        die_with_error("ERROR [upscale-zcu102::setup_video()] mmap failed: ", strerror(errno), p_res);
    }

    // Start video streaming
    printf("INFO [upscale-zcu102::setup_video()] Starting video streaming from v4l2\n");
    if (ioctl(p_res->v4l2_fd, VIDIOC_STREAMON, &p_res->v4l2_fmt.type) == -1) {
        die_with_error("ERROR [upscale-zcu102::setup_video()] Error starting video streaming: ", strerror(errno), p_res);
    }
}
*/
/**
 * Set up all things framebuffer related
 */
void setup_framebuffer(Resources *p_res){
    // Query framebuffer info
    if (ioctl(p_res->fb_dev_fd, FBIOGET_VSCREENINFO, &p_res->configurable_fb_info) == -1) {
        die_with_error("ERROR [upscale-zcu102::setup_framebuffer()] Error reading variable screen info: ", strerror(errno), p_res);
    }
    if (p_res->configurable_fb_info.bits_per_pixel != 16) {
        die_with_error("ERROR [upscale-zcu102::setup_framebuffer()] Framebuffer is not in RGB565 format\n", nullptr, p_res);
    }

    // Get fixed information
    if (ioctl(p_res->fb_dev_fd, FBIOGET_FSCREENINFO, &p_res->fixed_fb_info) == -1) {
        die_with_error("ERROR [upscale-zcu102::setup_framebuffer()] Error reading fixed screen info: ", strerror(errno), p_res);
    }
    printf("INFO [upscale-zcu102::setup_framebuffer()] Frame Buffer physical address: 0x%lx\n", p_res->fixed_fb_info.smem_start);
    printf("INFO [upscale-zcu102::setup_framebuffer()] Frame Buffer size:             0x%x\n", p_res->fixed_fb_info.smem_len);

    // Get variable screen information
    if (ioctl(p_res->fb_dev_fd, FBIOGET_VSCREENINFO, &p_res->configurable_fb_info) == -1) {
        die_with_error("ERROR [upscale-zcu102::setup_framebuffer()] Error reading variable screen info: ", strerror(errno), p_res);
    }
    
    // Map the framebuffer to a PhysMem object
    size_t fb_size = (p_res->configurable_fb_info.yres_virtual * p_res->configurable_fb_info.xres_virtual * p_res->configurable_fb_info.bits_per_pixel) / 8;
    printf("INFO [upscale-zcu102::setup_framebuffer()] Mapping framebuffer to PhysMem\n");
    
    p_res->fb_phys_base = p_res->fixed_fb_info.smem_start;
    p_res->fb_size_bytes = fb_size;
    p_res->fb_mem_block = PMM.alloc(p_res->fb_phys_base, p_res->fb_size_bytes);

    if(p_res->fb_mem_block == nullptr){
        die_with_error("ERROR [upscale-zcu102::setup_framebuffer()] Failed to map framebuffer to PhysMem\n", nullptr, p_res);
    }
}

/**
 * Compute the source address of every row in a TileInfo struct given the coordinates of the tile and PhysMem buffer.
 * Additionally, program the buffer descriptors for the MM2S channel and reset their complete bit.
 * If the given coordinates end up with the tile being out of bounds, the tile will be shifted to the left or up,
 * overlapping some with the previous tile.
 * @param src_block Pointer to the PhysMem object containing the source image
 * @param tile Pointer to the TileInfo struct containing tile x and y coordinates
 * @param xres The width of the source image in pixels
 * @param yres The height of the source image in pixels
 * @param bytes_pp The number of bytes per pixel in the source image
 * @param mm2s_bds Pointer to the MM2S buffer descriptor array to be programmed
 * @param src_offset Offset to start from src block. Default = 0
 * @return None
 */
void compute_tile_src_addr(PhysMem *src_block, TileInfo *tile, uint16_t xres, uint16_t yres, uint8_t bytes_pp, PhysMem** mm2s_bds, uint32_t src_offset=0){

    // Compute pixel coordinates of top left corner of tile
    uint32_t pixel_x = tile->tile_x * TILE_WIDTH_PIX;
    uint32_t pixel_y = tile->tile_y * TILE_HEIGHT_PIX;

    // Handle tiles not being fully in the image
    // 720: If pixel_x > 692, pixel_x = 692
    // 576: If pixel_y > 548, pixel_y = 548
    if(pixel_x > (xres - TILE_WIDTH_PIX))  pixel_x = xres - TILE_WIDTH_PIX;
    if(pixel_y > (yres - TILE_HEIGHT_PIX)) pixel_y = yres - TILE_HEIGHT_PIX;

    tile->src_pixel_x = pixel_x;
    tile->src_pixel_y = pixel_y;

    // Memory offset of the top left corner of the tile
    //                                Linear pixel number
    //                            vvvvvvvvvvvvvvvvvvvvvvvvvv
    uint32_t tile_start_offset = ((pixel_y * xres) + pixel_x) * bytes_pp;

    for(uint16_t row = 0; row < TILE_HEIGHT_PIX; row++){
        uint32_t row_offset = row * (xres * bytes_pp);
        tile->src_row_offset[row] = tile_start_offset + row_offset;

        tile->src_row_phys_addr[row] = tile->src_row_offset[row] + src_block->get_phys_address();

        ((BD_PTR)(mm2s_bds[row]->get_mem_ptr()))->buffer_address = tile->src_row_phys_addr[row];
        clear_cmplt_bit((BD_PTR)(mm2s_bds[row]->get_mem_ptr()));
    }
}

/**
 * Compute the destination address of every row in a TileInfo struct given a populated tile struct and PhysMem buffer. Additionally,
 * program DMA buffer descriptors for the S2MM channel and reset their complete bit.
 * The user MUST call compute_tile_src_addr before calling this function.
 * @param dst_block Where the DMA will write to
 * @param tile Pointer to the TileInfo struct - Must have source offsets populated by first calling compute_tile_src_offsets
 * @param xres_screen The width of the screen to write to (to compute row offsets)
 * @param upscale_factor The factor by which the image is being upscaled
 * @param bytes_pp The number of bytes per pixel in the source image
 * @param s2mm_bds Pointer to the S2MM buffer descriptor array to be programmed
 * @param xres_out The width of the output buffer (AKA the frame buffer) in pixels. 
 * @param start_x x coordinate on the screen of top left corner of frame
 * @param start_y y coordinate on the screen of top left corner of frame
 * @return 0 on success, -1 on error. Error if a write would occur before the frame buffer. TODO: add protection after fb
 */
int compute_tile_dst_addr(PhysMem* dst_block, TileInfo *tile, uint16_t xres_screen, uint32_t upscale_factor, uint8_t bytes_pp, PhysMem** s2mm_bds, \
    int start_x = 0, int start_y = 0){

    uint32_t dst_pixel_x = tile->src_pixel_x * upscale_factor;
    uint32_t dst_pixel_y = tile->src_pixel_y * upscale_factor;

    // Memory address of the top left corner of the tile
    int tile_start_offset = ((dst_pixel_y * (xres_screen)) + dst_pixel_x) * bytes_pp;

    tile_start_offset += (start_y * xres_screen + start_x) * bytes_pp;

    // Do not write into memory that is before the frame buffer
    if(tile_start_offset < 0) return -1;

    for(uint16_t row = 0; row < (TILE_HEIGHT_PIX * upscale_factor); row++){
        uint32_t row_offset = row * (xres_screen) * bytes_pp;
        tile->dst_row_offset[row] = tile_start_offset + row_offset;

        tile->dst_row_phys_addr[row] = tile->dst_row_offset[row] + dst_block->get_phys_address();

        ((BD_PTR)(s2mm_bds[row]->get_mem_ptr()))->buffer_address = tile->dst_row_phys_addr[row];
        clear_cmplt_bit((BD_PTR)(s2mm_bds[row]->get_mem_ptr()));
    }
    return 0;
}

/**
 * Initialize buffer descriptor chains for the MM2S and S2MM channels
 * @param mm2s Pointer to the MM2S buffer descriptor array
 * @param s2mm Pointer to the S2MM buffer descriptor array
 * @param mm2s_size The number of MM2S buffer descriptors
 * @param upscale_factor The factor by which the image is being upscaled
 * @param bytes_pp_in The number of bytes per pixel in the source image
 * @param bytes_pp_out The number of bytes per pixel in the destination image after passing thru HW
 * @return 0 on success, -1 on failure
 */
int init_buffer_descriptors(PhysMem **mm2s, PhysMem **s2mm, uint32_t mm2s_size, uint32_t upscale_factor, uint8_t bytes_pp_in, uint8_t bytes_pp_out){

    // Allocate physical memory for the buffer descriptors
    for(uint32_t i = 0; i < mm2s_size; i++){
        mm2s[i] = PMM.alloc(SG_BD_SIZE_BYTES);

        if(mm2s[i] == nullptr){
            printf("ERROR [upscale-zcu102::init_buffer_descriptors()] Failed to allocate mm2s buffer descriptor\n");
            return -1;
        }
        memset((void*)mm2s[i]->get_mem_ptr(), 0, SG_BD_SIZE_BYTES);
    }

    for(uint32_t i = 0; i < (mm2s_size * upscale_factor); i++){
        s2mm[i] = PMM.alloc(SG_BD_SIZE_BYTES);

        if(s2mm[i] == nullptr){
            printf("ERROR [upscale-zcu102::init_buffer_descriptors()] Failed to allocate s2mm buffer descriptor\n");
            return -1;
        }
        memset((void*)s2mm[i]->get_mem_ptr(), 0, SG_BD_SIZE_BYTES);
    }

    // Program mm2s to be linked lists
    // Set the SOF bit for the first BD
    // Set the buffer lengths
    // Set the next desc ptr and index
    // Set the EOF bit for the last BD
    set_sof_bit(((BD_PTR)mm2s[0]->get_mem_ptr()), 1);
    for(uint32_t i = 0; i < mm2s_size - 1; i++){
        BD_PTR current_bd = (BD_PTR)(mm2s[i]->get_mem_ptr());

        set_buffer_length(current_bd, TILE_WIDTH_PIX * bytes_pp_in);
        current_bd->next_desc_index = i + 1;
        current_bd->next_desc_ptr = mm2s[i+1]->get_phys_address();
    }
    BD_PTR last_mm2s = (BD_PTR)(mm2s[mm2s_size - 1]->get_mem_ptr());
    set_eof_bit(last_mm2s, 1);
    set_buffer_length(last_mm2s, TILE_WIDTH_PIX * bytes_pp_in);
    last_mm2s->next_desc_ptr = mm2s[0]->get_phys_address();
    last_mm2s->next_desc_index = 0;

    // Same for s2mm
    set_sof_bit(((BD_PTR)s2mm[0]->get_mem_ptr()), 1);
    for(uint32_t i = 0; i < (mm2s_size * upscale_factor) - 1; i++){
        BD_PTR current_bd = (BD_PTR)(s2mm[i]->get_mem_ptr());

        set_buffer_length(current_bd, TILE_WIDTH_PIX * upscale_factor * bytes_pp_out);
        current_bd->next_desc_index = i + 1;
        current_bd->next_desc_ptr = s2mm[i+1]->get_phys_address();
    }
    BD_PTR last_s2mm = (BD_PTR)(s2mm[(mm2s_size * upscale_factor) - 1]->get_mem_ptr());
    set_eof_bit(last_s2mm, 1);
    set_buffer_length(last_s2mm, TILE_WIDTH_PIX * upscale_factor * bytes_pp_out);
    last_s2mm->next_desc_ptr = s2mm[0]->get_phys_address();
    last_s2mm->next_desc_index = 0;

    return 0;
}

void draw_dots(PhysMem* block, TileInfo *tile, Resources* res, uint8_t r, uint8_t g, uint8_t b){
    uint32_t tile_offset = tile->dst_row_offset[0];
    if(block->write_byte(tile_offset+0, r) < 0) die_with_error("Error writing byte", nullptr, res);
    if(block->write_byte(tile_offset+1, g) < 0) die_with_error("Error writing byte", nullptr, res);
    if(block->write_byte(tile_offset+2, b) < 0) die_with_error("Error writing byte", nullptr, res);
}

void draw_outline(PhysMem* block, TileInfo *tile, Resources* res, uint8_t r, uint8_t g, uint8_t b){
    // Draw the first and last rows
    uint8_t pixel[] = {r, g, b};

    // Draw the top and bottom rows
    for(uint16_t i = 0; i < (TILE_WIDTH_PIX * UPSCALE_FACTOR * 3); i++){
        if(block->write_byte(tile->dst_row_offset[0] + i, pixel[i % 3]) < 0) 
            die_with_error("draw_outline: Error writing byte", nullptr, res);
        if(block->write_byte(tile->dst_row_offset[(TILE_HEIGHT_PIX * UPSCALE_FACTOR) - 1] + i, pixel[i % 3]) < 0) 
            die_with_error("draw_outline: Error writing byte", nullptr, res);
    }

    // Draw the first and last columns
    for(uint32_t i = 0; i < (TILE_HEIGHT_PIX * UPSCALE_FACTOR); i++){
        if(block->write_byte(tile->dst_row_offset[i], r) < 0) 
            die_with_error("draw_outline: Error writing byte", nullptr, res);
        if(block->write_byte(tile->dst_row_offset[i] + 1, g) < 0) 
            die_with_error("draw_outline: Error writing byte", nullptr, res);
        if(block->write_byte(tile->dst_row_offset[i] + 2, b) < 0) 
            die_with_error("draw_outline: Error writing byte", nullptr, res);

        if(block->write_byte(tile->dst_row_offset[i] + (TILE_WIDTH_PIX * UPSCALE_FACTOR * 3) - 3, r) < 0) 
            die_with_error("draw_outline: Error writing byte", nullptr, res);
        if(block->write_byte(tile->dst_row_offset[i] + (TILE_WIDTH_PIX * UPSCALE_FACTOR * 3) - 2, g) < 0) 
            die_with_error("draw_outline: Error writing byte", nullptr, res);
        if(block->write_byte(tile->dst_row_offset[i] + (TILE_WIDTH_PIX * UPSCALE_FACTOR * 3) - 1, b) < 0) 
            die_with_error("draw_outline: Error writing byte", nullptr, res);
    }
}

/**
 * Save the contents of before frame and after frame to a file. Assumes input and output are RGB888
 * @param before_frame Pointer to the before frame
 * @param after_frame Pointer to the after frame
 * @return 0 on success, -1 on failure
 */
int save_screenshot(void *before_frame, void* after_frame){
    // Save the input888 block and interp 888 block to a file
    FILE *input_fp = fopen("input888.raw", "wb");
    if(input_fp == nullptr){
        printf("ERROR [upscale-zcu102::save_screenshot()] Failed to open input888.raw for writing\n");
        return -1;
    }
    size_t num_bytes = fwrite(before_frame, 1, INPUT_VIDEO_WIDTH*INPUT_VIDEO_HEIGHT*3, input_fp);
    if(num_bytes != INPUT_VIDEO_WIDTH*INPUT_VIDEO_HEIGHT*3){
        printf("ERROR [upscale-zcu102::save_screenshot()] Failed to write input888.raw\n");
        return -1;
    }
    fflush(input_fp);
    fclose(input_fp);

    FILE *interp_fp = fopen("interp888.raw", "wb");
    if(interp_fp == nullptr){
        printf("ERROR [upscale-zcu102::save_screenshot()] Failed to open interp888.raw for writing\n");
        return -1;
    }
    num_bytes = fwrite(after_frame, 1, INPUT_VIDEO_WIDTH*INPUT_VIDEO_HEIGHT*3*UPSCALE_FACTOR*UPSCALE_FACTOR, interp_fp);
    if(num_bytes != INPUT_VIDEO_WIDTH*INPUT_VIDEO_HEIGHT*3*UPSCALE_FACTOR*UPSCALE_FACTOR){
        printf("ERROR [upscale-zcu102::save_screenshot()] Failed to write interp888.raw\n");
        return -1;
    }
    fflush(interp_fp);
    fclose(interp_fp);

    return 0;
}

/**
 * Save the contents of before frame and after frame to a file. Assumes input and output are RGB888
 * @param before_frame Pointer to the before frame
 * @param after_frame Pointer to the after frame
 * @return 0 on success, -1 on failure
 */
int save_yuyv_888_screenshot(void *yuyv, void* rgb888){
    // Save the input888 block and interp 888 block to a file
    FILE *input_fp = fopen("input_yuyv.raw", "wb");
    if(input_fp == nullptr){
        printf("ERROR [upscale-zcu102::save_screenshot()] Failed to open input_yuyv.raw for writing\n");
        return -1;
    }
    size_t num_bytes = fwrite(yuyv, 1, INPUT_VIDEO_WIDTH*INPUT_VIDEO_HEIGHT*2, input_fp);
    if(num_bytes != INPUT_VIDEO_WIDTH*INPUT_VIDEO_HEIGHT*2){
        printf("ERROR [upscale-zcu102::save_screenshot()] Failed to write input_yuyv.raw\n");
        return -1;
    }
    fflush(input_fp);
    fclose(input_fp);

    FILE *interp_fp = fopen("rgb888.raw", "wb");
    if(interp_fp == nullptr){
        printf("ERROR [upscale-zcu102::save_screenshot()] Failed to open rgb888.raw for writing\n");
        return -1;
    }
    num_bytes = fwrite(rgb888, 1, INPUT_VIDEO_WIDTH*INPUT_VIDEO_HEIGHT*3, interp_fp);
    if(num_bytes != INPUT_VIDEO_WIDTH*INPUT_VIDEO_HEIGHT*3){
        printf("ERROR [upscale-zcu102::save_screenshot()] Failed to write rgb888.raw\n");
        return -1;
    }
    fflush(interp_fp);
    fclose(interp_fp);

    return 0;
}

int main(int argc, char *argv[]){
    printf("INFO [upscale-zcu102] Entering main\n");
    
    Resources resources;

    argparse::ArgumentParser parser("sg-interpolate2x");
    // parser.add_argument("-vid", "--video_device").help("Video device to read frames from").default_value("/dev/video0");
    parser.add_argument("-fbd", "--fb_device").help("Framebuffer device to write frames to").default_value("/dev/fb0");
    parser.add_argument("-x").help("Starting x-coordinate of frame on screen").scan<'d', int>().default_value(0);
    parser.add_argument("-y").help("Starting y-coordinate of frame on screen").scan<'d', int>().default_value(0);
    parser.add_argument("--dots").help("Flag to show start of a tile with a red dot").flag();
    parser.add_argument("--lines").help("Flag to draw an outline around all tiles").flag(); // .flag = .default_value(false).implicit_value(true);
    parser.add_argument("--no_self_test").help("Flag to skip the DMA self test").flag();
    parser.add_argument("--double_test").help("Flag to run the self test twice").flag();
    parser.add_argument("--no_fps").help("Flag to disable printing fps").flag();
    parser.add_argument("--conv").help("Override to send only to conv").flag();
    parser.add_argument("--interp").help("Override to send only to interp").flag();
    parser.add_argument("--screenshot").help("Set this flag to take a single screenshot of the before frame and after frame").flag();

    try{
        parser.parse_args(argc, argv);
    }
    catch(const std::runtime_error& err){
        std::cout << err.what() << std::endl;
        return -1;
    }

    // std::string video_dev_str = parser.get<std::string>("--video_device");
    std::string fb_dev_str = parser.get<std::string>("--fb_device");

    // const char *video_device = video_dev_str.c_str();
    const char *fb_device = fb_dev_str.c_str();

    // printf("INFO [upscale-zcu102::main()] Video device: %s\n", video_device);
    printf("INFO [upscale-zcu102::main()] Framebuffer device: %s\n", fb_device);

    if(parser["--dots"] == true){
        printf("INFO [upscale-zcu102] Drawing red dots at start of each tile\n");
    }
    else if(parser["--lines"] == true){
        printf("INFO [upscale-zcu102] Drawing outline around all tiles\n");
    }
    if(parser["--no_self_test"] == true){
        printf("INFO [upscale-zcu102] Skipping DMA self test\n");
    }

    // Register CTRL-C handler
    signal(SIGINT, sigint_handler);

    // Open /dev/mem, video device, framebuffer device
    // Initialize Physical Memory Manager
    // Initialize input888 buffer, interpolated888 buffer
    init_resources(&resources, fb_device);

    // Setup video reosources
    // setup_video(&resources);

    // Setup framebuffer resources
    setup_framebuffer(&resources);

	//axi_dma1 configured as NOT scatter gather, MM2S --> FIFO --> S2MM
	AXIDMA dma1(DMA_0_AXI_LITE_BASE, resources.dev_mem_fd);
	printf("INFO [upscale-zcu102] Created DMA object with base address 0x%08X\n", DMA_0_AXI_LITE_BASE);
	if(dma1.initialize() != 0){
        die_with_error("ERROR [upscale-zcu102] Failed to initialize DMA object\n", nullptr, &resources);
	}

    // Run the self test twice in a row, only if upscale factor is 1
    if(parser["--no_self_test"] == false && (UPSCALE_FACTOR==1)){
        if(dma1.self_test() < 0){
            die_with_error("ERROR [upscale-zcu102] DMA self test failed\n", nullptr, &resources);
        }
        if(parser["--double_test"] == true){
            if(dma1.self_test() < 0){
                die_with_error("ERROR [upscale-zcu102] DMA self test failed\n", nullptr, &resources);
            }
        }
    }

    // Instantiate variance block
    VarianceDispatcher var(VARIANCE_BASE_ADDR, resources.dev_mem_fd);
    puts("INFO [upscale-zcu102] Initializing VarianceDispatcher\n");
    if(var.init() != 0){
        die_with_error("ERROR [upscale-zcu102] Failed to initialize Variance Dispatcher\n", nullptr, &resources);
    }

    printf("INFO [upscale-zcu102] Variance ready bit: %d\n", var.is_ready());
    var.start();
    var.enable_auto_restart();
    if(parser["--conv"] == true){
        printf("INFO [upscale-zcu102] Overriding to convolution only\n");
        var.set_override(VARIARNCE_OVERRIDE_MODE_CONV);
    }
    else if(parser["--interp"] == true){
        printf("INFO [upscale-zcu102] Overriding to interpolation only\n");
        var.set_override(VARIARNCE_OVERRIDE_MODE_INTERP);
    }

    // Initialize the output buffer to cyan for debugging purposes
    // for(uint32_t i = 0; i < resources.interp888_block->size() / 3; i++){
    //     ((uint8_t*)(resources.interp888_block->get_mem_ptr()))[i * 3] = 0;
    //     ((uint8_t*)(resources.interp888_block->get_mem_ptr()))[i * 3 + 1] = 0x8F;
    //     ((uint8_t*)(resources.interp888_block->get_mem_ptr()))[i * 3 + 2] = 0x8F;
    // }

    for(uint32_t i = 0; i < resources.input888_block->size() / 4; i++){
        // YUYV Cyan --> 0x00B2AAB2 (don't ask me how, it's from chatgpt)
        ((uint32_t*)(resources.input888_block->get_mem_ptr()))[i] = 0x00B2AAB2;
    }
    
    uint32_t frame_loop_count = 0;
    unsigned long start_jiffies = get_jiffies();

    TileInfo tile;
    // Note: Frame is 720x576. However, game data does not start until y=76, and only goes until y=496 (call it 72 and 512)
    uint16_t num_vert_tiles = INPUT_VIDEO_HEIGHT / TILE_HEIGHT_PIX;
    uint16_t num_horz_tiles = INPUT_VIDEO_WIDTH  / TILE_WIDTH_PIX;
        
    // Handle partial tiles
    if((INPUT_VIDEO_HEIGHT % TILE_HEIGHT_PIX) != 0) num_vert_tiles++;
    if((INPUT_VIDEO_WIDTH  % TILE_WIDTH_PIX)  != 0) num_horz_tiles++;

    // Initialize scatter gather buffer descriptors for tiles
    dma1.reset_dma();
    PhysMem* mm2s_bds[TILE_HEIGHT_PIX];
    PhysMem* s2mm_bds[TILE_HEIGHT_PIX * UPSCALE_FACTOR];
    // if(init_buffer_descriptors(mm2s_bds, s2mm_bds, TILE_HEIGHT_PIX, UPSCALE_FACTOR, 3, 3) < 0){
    if(init_buffer_descriptors(mm2s_bds, s2mm_bds, TILE_HEIGHT_PIX, UPSCALE_FACTOR, 2, 2) < 0){
        die_with_error("ERROR [upscale-zcu102] Failed to initialize buffer descriptors\n", nullptr, &resources);
    }
    BD_PTR last_mm2s_bd = (BD_PTR)(mm2s_bds[TILE_HEIGHT_PIX - 1]->get_mem_ptr());
    BD_PTR last_s2mm_bd = (BD_PTR)(s2mm_bds[(TILE_HEIGHT_PIX * UPSCALE_FACTOR) - 1]->get_mem_ptr());

    dma1.set_mm2s_curdesc(mm2s_bds[0]->get_phys_address());
    dma1.set_s2mm_curdesc(s2mm_bds[0]->get_phys_address());
    dma1.start_mm2s();
    dma1.start_s2mm();

    printf("INFO [upscale-zcu102] Starting video output\n");

    // Queue the first buffer, then alternate
    // if (ioctl(resources.v4l2_fd, VIDIOC_QBUF, &resources.v4l2_frame_bufs[0]) == -1) {
    //     die_with_error("ERROR [upscale-zcu102] Error queueing buffer\n", nullptr, &resources);
    // }

    // Open up the image and copy it to input888 block
    // Open the file "test_yuyv_720_576.bin" and read 200 bytes into resources.input888_block->get_mem_ptr()
    FILE *file = fopen("example_yuyv_img.bin", "rb");
    if (file == nullptr) {
        die_with_error("ERROR [upscale-zcu102] Failed to open test_yuyv_720_576.bin\n", strerror(errno), &resources);
    }
    size_t expected = INPUT_VIDEO_HEIGHT*INPUT_VIDEO_WIDTH*2;
    size_t bytes_read = fread((void*)resources.input888_block->get_mem_ptr(), 1, INPUT_VIDEO_HEIGHT*INPUT_VIDEO_WIDTH*2, file);
    if (bytes_read != expected) {
        die_with_error("ERROR [upscale-zcu102] Failed to read bytes from example_yuyv_img.bin\n", nullptr, &resources);
    }
    fclose(file);

    int queued_buffer = 0;
    int next_buffer   = 1;
    while(!die_flag){
        
        // Capture a frame
        // if (ioctl(resources.v4l2_fd, VIDIOC_QBUF, &resources.v4l2_frame_bufs[next_buffer]) == -1) {
        //     die_with_error("ERROR [upscale-zcu102] Error queueing buffer\n", nullptr, &resources);
        // }
        // if (ioctl(resources.v4l2_fd, VIDIOC_DQBUF, &resources.v4l2_frame_bufs[queued_buffer]) == -1) {
        //     die_with_error("ERROR [upscale-zcu102] Error dequeueing buffer: ", strerror(errno), &resources);
        // }
        
        /*
        yuyv_to_rgb888((uint8_t*)(resources.vid_mem_ptr[queued_buffer]), 
        (uint8_t*)(resources.input888_block->get_mem_ptr()), 
        INPUT_VIDEO_WIDTH, INPUT_VIDEO_HEIGHT);
        */
        // memcpy((void*)resources.input888_block->get_mem_ptr(), resources.vid_mem_ptr[queued_buffer], INPUT_VIDEO_HEIGHT*INPUT_VIDEO_WIDTH*2);

        // save_yuyv_888_screenshot((void*)resources.vid_mem_ptr[queued_buffer], (void*)(resources.input888_block->get_mem_ptr()));
        
        // Next buffer now has the actual data
        if(queued_buffer == 0){
            queued_buffer = 1;
            next_buffer = 0;
        } else{
            queued_buffer = 0;
            next_buffer = 1;
        }

        // Convert v4l2 frame to RGB888
        // Loop
        //  - Send a tile down to the PL for interpolation
        //  - Receive the tile in the interp_888 buffer / PMM block
        // Move to framebuffer, converting to RGB565 on the way

        unsigned long before_tile_jiffies = get_jiffies();
        for(uint16_t tx = 0; tx < num_horz_tiles; tx++){
            for(uint16_t ty = 2; ty < num_vert_tiles - 2; ty++){
                
                // Calculate physical addresses for each row of the tile
                // Set the BD rings appropriately
                // Send the tile down to the PL for interpolation
                // Calculate destination addresses for each row of the tile
                // Receive the tile in the interp_888 buffer / PMM block
                tile.tile_x = tx;
                tile.tile_y = ty;
                // compute_tile_src_addr(resources.input888_block, &tile, INPUT_VIDEO_WIDTH, INPUT_VIDEO_HEIGHT, 3, mm2s_bds, 0);
                compute_tile_src_addr(resources.input888_block, &tile, INPUT_VIDEO_WIDTH, INPUT_VIDEO_HEIGHT, 2, mm2s_bds, 0);
                
                // Start the MM2S transfer
                dma1.set_mm2s_taildesc(mm2s_bds[TILE_HEIGHT_PIX - 1]->get_phys_address());

                // Calculate destination addresses for each row of the tile
                // compute_tile_dst_addr(resources.interp888_block, &tile, INPUT_VIDEO_WIDTH, UPSCALE_FACTOR, 3, s2mm_bds);

                int result = compute_tile_dst_addr(resources.fb_mem_block, &tile, resources.configurable_fb_info.xres_virtual,\
                    UPSCALE_FACTOR, 2, s2mm_bds, parser.get<int>("-x"), parser.get<int>("-y"));

                // if we try to write before the frame
                if(result < 0) continue;

                // Wait for MM2S transfer to complete
                if(dma1.poll_bd_cmplt(last_mm2s_bd, DMA_SYNC_TRIES) < 0) {
                    dma1.print_debug_info();
                    // printf("INFO [upscale-zcu102] DMA total MM2S bytes: %d\n", dma1.get_total_bytes_mm2s());
                    die_with_error("ERROR [upscale-zcu102] MM2S DMA transfer timed out\n", nullptr, &resources);
                }
                // Start the S2MM transfer
                dma1.set_s2mm_taildesc(s2mm_bds[(TILE_HEIGHT_PIX * UPSCALE_FACTOR) - 1]->get_phys_address());
                if(dma1.poll_bd_cmplt(last_s2mm_bd, DMA_SYNC_TRIES) < 0){
                    dma1.print_debug_info();
                    // printf("INFO [upscale-zcu102] DMA total S2MM bytes: %d\n", dma1.get_total_bytes_s2mm());
                    die_with_error("ERROR [upscale-zcu102] S2MM DMA transfer timed out\n", nullptr, &resources);
                } 

                // Set a pixel at the top left corner of the tile to red
                // TODO: need to fix to operate on RGB565
                if(parser["--dots"] == true){
                    // draw_dots(resources.interp888_block, &tile, &resources, 0xFF, 0, 0);
                    draw_dots(resources.fb_mem_block, &tile, &resources, 0xFF, 0, 0);
                }
                else if(parser["--lines"] == true){
                    // draw_outline(resources.interp888_block, &tile, &resources, 0x00, 0xFF, 0);
                    draw_outline(resources.fb_mem_block, &tile, &resources, 0x00, 0xFF, 0);
                }

                static bool first_tile = true;
                if(first_tile){
                    first_tile = false;
                    printf("INFO [upscale-zcu102] First tile successfully processed\n");
                }
            }
        }
        unsigned long after_tile_jiffies = get_jiffies();
        static bool first = true;
        if(first){
            float elapsed_time = (float)(after_tile_jiffies - before_tile_jiffies) / (float)sysconf(_SC_CLK_TCK);
            printf("INFO [upscale-zcu102] Time to process tiles: %0.5f seconds\n", elapsed_time);
            printf("INFO [upscale-zcu102] Number of jiffies: %lu\n", after_tile_jiffies - before_tile_jiffies);
            first = false;
        }

        static bool screenshot_taken = false;
        if(!screenshot_taken && (parser["--screenshot"] == true)){
            screenshot_taken = true;
            printf("INFO [upscale-zcu102] Taking screenshot...\n");
            int ss = save_screenshot((void*)resources.input888_block->get_mem_ptr(), (void*)resources.interp888_block->get_mem_ptr());
            if(ss < 0){
                die_with_error("ERROR [upscale-zcu102] Failed to save screenshot\n", nullptr, &resources);
            }
            printf("INFO [upscale-zcu102] Screenshot saved\n");
        }

        // Calculate frame rate
        frame_loop_count++;
        if((frame_loop_count % 100) == 99){
            unsigned long end_jiffies = get_jiffies();
            unsigned long elapsed_jiffies = end_jiffies - start_jiffies;
            unsigned long jiffies_per_sec = sysconf(_SC_CLK_TCK);

            float fps = (float)frame_loop_count / ((float)elapsed_jiffies / (float)jiffies_per_sec);

            frame_loop_count = 0;
            start_jiffies = end_jiffies;
            printf("INFO [upscale-zcu102] FPS: %0.3f\n", fps);
        }    
    } // End of while loop

    // Stop video streaming
    // if (ioctl(resources.v4l2_fd, VIDIOC_STREAMOFF, &resources.v4l2_fmt.type) == -1) {
    //     die_with_error("ERROR [upscale-zcu102] Error stopping video streaming: ", strerror(errno), &resources);
    // }

    // Cleanup
    cleanup_resources(&resources);

	return 0;
}
