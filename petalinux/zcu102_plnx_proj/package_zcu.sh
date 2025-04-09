#!/bin/bash
echo "INFO [package.sh] Sourcing petalinux settings"

if source /tools/Xilinx/PetaLinux/2023.1/settings.sh ; then
	echo "INFO [package_zcu.sh] Successfully sourced petalinux settings"
else
	echo "ERROR [package_zcu.sh] Error sourcing build tools"
	exit 1
fi

rm ./images/linux/petalinux-sdimage.wic
rm ./images/linux/petalinux-sdimage.wic.gz

if petalinux-package --boot --fsbl ./images/linux/zynqmp_fsbl.elf --u-boot --force ; then
	echo "INFO [package_zcu.sh] Successfully packaged petalinux boot image"
else
	echo "ERROR [package_zcu.sh] Error packaging petalinux boot image"
	exit 1
fi

if petalinux-package --wic --wks ./zcu102_wks.wks  --bootfiles "BOOT.BIN ramdisk.cpio.gz.u-boot boot.scr Image system.dtb" ; then
	echo "INFO [package_zcu.sh] Successfully packaged petalinux image"
else
	echo "ERROR [package_zcu.sh] Error packaging petalinux image"
	exit 1
fi

echo "INFO [package_zcu.sh] Zipping petalinux-sdimage.wic..."
gzip -f -k -v ./images/linux/petalinux-sdimage.wic
echo "INFO [package_zcu.sh] Finished"
