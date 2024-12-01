#!/bin/bash

echo "INFO [build_and_package.sh] Sourcing petalinux settings"

if source /tools/Xilinx/PetaLinux/2023.1/settings.sh ; then
	echo "INFO [build_and_package.sh] Successfully sourced petalinux settings"
else
	echo "ERROR [build_and_package.sh] Error sourcing build tools"
	exit 1
fi

echo "INFO [build_and_package.sh] Doing full clean of build"

if petalinux-build -x mrproper ; then
	echo "INFO [build_and_package.sh] Successfully cleaned build"
else
	echo "ERROR [build_and_package.sh] Error cleaning build"
	exit 1
fi

echo "INFO [build_and_package.sh] Building petalinux"

if petalinux-build ; then
	echo "INFO [build_and_package.sh] Successfully built petalinux"
else
	echo "ERROR [build_and_package.sh] Error building petalinux"
	exit 1
fi

echo "INFO [build_and_package.sh] Packaging petalinux image"

if petalinux-package --wic --wks ./kv260_wks.wks  --bootfiles "ramdisk.cpio.gz.u-boot boot.scr Image system.dtb" ; then
	echo "INFO [build_and_package.sh] Successfully packaged petalinux image"
else
	echo "ERROR [build_and_package.sh] Error packaging petalinux image"
	exit 1
fi

if petalinux-package --boot --fsbl ./images/linux/zynqmp_fsbl.elf --fpga ./images/linux/system.bit --u-boot --force ; then
	echo "INFO [build_and_package.sh] Successfully packaged petalinux boot image"
else
	echo "ERROR [build_and_package.sh] Error packaging petalinux boot image"
	exit 1
fi

echo "INFO [do_build_and_package.sh] Zipping petalinux-sdimage.wic..."
gzip -f -k -v ./images/linux/petalinux-sdimage.wic
echo "INFO [do_build_and_package.sh] Finished"
