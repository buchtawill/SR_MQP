#include <stdio.h>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <unistd.h>

int main(int argc, char *argv[]) {

	if(argc < 2) {
		printf("Usage: %s <fb_device>\n", argv[0]);
		printf("Example: sudo %s /dev/fb0\n", argv[0]);
		return 1;
	}

    const char *fb_device = argv[1];
    int fb_fd = open(fb_device, O_RDWR);
    if (fb_fd == -1) {
        perror("Opening framebuffer device");
        return 1;
    }

    struct fb_var_screeninfo vinfo;

    // Get variable screen information
    if (ioctl(fb_fd, FBIOGET_VSCREENINFO, &vinfo) == -1) {
        perror("Getting variable screen info");
        close(fb_fd);
        return 1;
    }

    printf("Framebuffer Information:\n");
    printf("Resolution: %dx%d\n", vinfo.xres, vinfo.yres);
    printf("Virtual Resolution: %dx%d\n", vinfo.xres_virtual, vinfo.yres_virtual);
    printf("Bits Per Pixel: %d\n", vinfo.bits_per_pixel);
    printf("Pixel Format:\n");
    printf("  Red: Offset=%d, Length=%d\n", vinfo.red.offset, vinfo.red.length);
    printf("  Green: Offset=%d, Length=%d\n", vinfo.green.offset, vinfo.green.length);
    printf("  Blue: Offset=%d, Length=%d\n", vinfo.blue.offset, vinfo.blue.length);
    printf("  Transparency: Offset=%d, Length=%d\n", vinfo.transp.offset, vinfo.transp.length);

    close(fb_fd);
    return 0;
}