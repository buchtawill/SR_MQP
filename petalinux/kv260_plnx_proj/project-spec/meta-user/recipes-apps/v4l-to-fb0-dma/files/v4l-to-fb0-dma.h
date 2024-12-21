


#ifndef V4L_TO_FB0_DMA_H_
#define V4L_TO_FB0_DMA_H_

#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <linux/fb.h>
#include <stdint.h>

#define INPUT_VIDEO_WIDTH        720
#define INPUT_VIDEO_HEIGHT       576
#define INPUT_VIDEO_PIXEL_FORMAT V4L2_PIX_FMT_YUYV

typedef struct {
    int  dev_mem_fd;
    int  v4l2_fd;
    int  fb_dev_fd;
    void *vid_mem_phys;
    void *fb_mem_phys;
    uint16_t *fb_mem_pix;
    void *rgb_565_buf_phys;
    uint16_t *rgb_565_buf_pix;

    struct fb_var_screeninfo configurable_fb_info;
    struct fb_fix_screeninfo fixed_fb_info;

    struct v4l2_format v4l2_fmt;
    struct v4l2_requestbuffers v4l2_req;
    struct v4l2_buffer v4l2_frame_buf;

    uint32_t fb_phys_base;
    uint32_t fb_size_bytes;
    uint32_t vid_mem_phys_base;
    uint32_t vid_mem_size_bytes;
    uint32_t rgb_565_buf_phys_base;
    uint32_t rgb_565_buf_size_bytes;

} Resources;


void cleanup_resources(Resources *p_res);


#endif // V4L_TO_FB0_DMA_H_