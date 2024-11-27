#!/bin/bash
echo "INFO [package.sh] Sourcing petalinux settings"

if petalinux-package --wic --wks ./kv260_wks.wks  --bootfiles "ramdisk.cpio.gz.u-boot boot.scr Image system.dtb" ; then
	echo "INFO [package.sh] Successfully packaged petalinux image"
else
	echo "ERROR [package.sh] Error packaging petalinux image"
	exit 1
fi

if petalinux-package --boot --fsbl ./images/linux/zynqmp_fsbl.elf --fpga ./images/linux/system.bit --u-boot --force ; then
	echo "INFO [package.sh] Successfully packaged petalinux boot image"
else
	echo "ERROR [package.sh] Error packaging petalinux boot image"
	exit 1
fi