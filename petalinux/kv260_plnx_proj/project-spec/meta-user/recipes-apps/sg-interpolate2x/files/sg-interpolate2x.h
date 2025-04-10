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
#define VARIANCE_BASE_ADDR		0xA0000000

#define INPUT_VIDEO_WIDTH       720
#define INPUT_VIDEO_HEIGHT      576
#define INPUT_VIDEO_PIXEL_FMT   V4L2_PIX_FMT_YUYV

#define RGB565_BUF_SIZE_BYTES   (INPUT_VIDEO_WIDTH * INPUT_VIDEO_HEIGHT * 2) // 2 bytes per pixel

#define UPSCALE_FACTOR          1
#define TILE_WIDTH_PIX          ((uint32_t)32)
#define TILE_HEIGHT_PIX         ((uint32_t)32)

/**
 * Contains relevant information about a given image tile.
 */
typedef struct {
    
    // Memory offset (in bytes) of the start of a row. Relative to address of top left pixel of whole image
    uint32_t src_row_offset[TILE_HEIGHT_PIX];

    // Memory address of the start of a row.
    uint32_t src_row_phys_addr[TILE_HEIGHT_PIX];

    // Memory address of the start of a row. Relative to address of top left pixel of whole image
    uint32_t dst_row_offset[TILE_HEIGHT_PIX * UPSCALE_FACTOR];

    // Memory address of the start of a row.
    uint32_t dst_row_phys_addr[TILE_HEIGHT_PIX * UPSCALE_FACTOR];

    uint16_t tile_x; // Requested x tile coordinate (in units of tiles)
    uint16_t tile_y; // Requested y tile coordinate (in units of tiles)

    uint16_t src_pixel_x; // Actual pixel x coordinate in source image, taking bounds into account
    uint16_t src_pixel_y; // Actual pixel y coordinate in source image, taking bounds into account

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
    // PhysMem *vid_mem_blocks[2];
    void *vid_mem_ptr[2];
    uint32_t vid_mem_size_bytes;
    struct v4l2_format v4l2_fmt;
    struct v4l2_requestbuffers v4l2_req;
    struct v4l2_buffer v4l2_frame_bufs[2];

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
