/**
 * This file contains various definitions for the v4l-to-fb0-dma application
 */


#ifndef INTERPOLATE2X_H_
#define INTERPOLATE2X_H_

#include <linux/videodev2.h>    // V4L2 API
#include <linux/fb.h>           // Framebuffer API
#include <stdint.h> 
#include "PhysMem.h"

// Addresses found in SR_MQP/petalinux/kv260_plnx_proj/components/plnx_workspace/device-tree/device-tree/pl.dtsi
// Or in the vivado block diagram address editor
#define DMA_0_AXI_LITE_BASE		0xA0020000

#define INPUT_VIDEO_WIDTH       720
#define INPUT_VIDEO_HEIGHT      576
#define INPUT_VIDEO_PIXEL_FMT   V4L2_PIX_FMT_YUYV

#define RGB565_BUF_SIZE_BYTES   (INPUT_VIDEO_WIDTH * INPUT_VIDEO_HEIGHT * 2) // 2 bytes per pixel

#define UPSCALE_FACTOR          2
#define TILE_WIDTH_PIX          28
#define TILE_HEIGHT_PIX         28

/**
 * Contains relevant information about a given image tile.
 */
typedef struct {
    
    // Physical address of each row in the source buffer
    uint32_t src_row_addr[TILE_HEIGHT_PIX];

    // Physical address of each row in the destination buffer
    uint32_t dst_row_addr[TILE_HEIGHT_PIX * UPSCALE_FACTOR];

    // Pointer to each row from the source buffer
    uint8_t* src_row_ptr[TILE_HEIGHT_PIX];

    // Pointer to each row in the destination buffer
    uint8_t* dst_row_ptr[TILE_HEIGHT_PIX * UPSCALE_FACTOR];

    uint16_t  tile_x; // x pixel coordinate of the tile
    uint16_t  tile_y; // y pixel coordinate of the tile

}TileInfo;

/**
 * This struct contains all resources that the main program uses.
 */
typedef struct {

    // File descriptors
    int  dev_mem_fd;
    int  v4l2_fd;
    int  fb_dev_fd;

    // Video resources
    PhysMem *vid_mem_block;
    uint32_t vid_mem_phys_base;
    uint32_t vid_mem_size_bytes;
    struct v4l2_format v4l2_fmt;
    struct v4l2_requestbuffers v4l2_req;
    struct v4l2_buffer v4l2_frame_buf;

    // Framebuffer resources
    PhysMem *fb_mem_block;
    uint16_t *fb_mem_pix;
    uint32_t fb_phys_base;  // Easier to keep track of the physical address
    uint32_t fb_size_bytes; // Easier to keep track of the size in bytes
    struct fb_var_screeninfo configurable_fb_info;
    struct fb_fix_screeninfo fixed_fb_info;

    // Various pixel buffer blocks
    PhysMem *input888_block;
    PhysMem *interp888_block;

} Resources;

void cleanup_resources(Resources *p_res);

#endif // INTERPOLATE2X_H_
