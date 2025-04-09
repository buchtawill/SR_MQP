#!/bin/bash

UNIX_TIME_START=$(date +%s)
echo "INFO [do_build_and_package_zcu.sh] Starting petalinux build and package process"
echo "INFO [do_build_and_package_zcu.sh] Sourcing petalinux settings"

if source /tools/Xilinx/PetaLinux/2023.1/settings.sh ; then
	echo "INFO [do_build_and_package_zcu.sh] Successfully sourced petalinux settings"
else
	echo "ERROR [do_build_and_package_zcu.sh] Error sourcing build tools"
	exit 1
fi

echo "INFO [do_build_and_package_zcu.sh] Doing full clean of build"

if petalinux-build -x mrproper ; then
	echo "INFO [do_build_and_package_zcu.sh] Successfully cleaned build"
else
	echo "ERROR [do_build_and_package_zcu.sh] Error cleaning build"
	exit 1
fi

echo "INFO [do_build_and_package_zcu.sh] Building petalinux"

if petalinux-build ; then
	echo "INFO [do_build_and_package_zcu.sh] Successfully built petalinux"
else
	echo "ERROR [do_build_and_package_zcu.sh] Error building petalinux"
	exit 1
fi

echo "INFO [do_build_and_package_zcu.sh] Packaging petalinux image"

if petalinux-package --boot --fsbl ./images/linux/zynqmp_fsbl.elf --u-boot --force ; then
	echo "INFO [do_build_and_package_zcu.sh] Successfully packaged petalinux boot image"
else
	echo "ERROR [do_build_and_package_zcu.sh] Error packaging petalinux boot image"
	exit 1
fi

if petalinux-package --wic --wks ./zcu102_wks.wks  --bootfiles "BOOT.BIN ramdisk.cpio.gz.u-boot boot.scr Image system.dtb" ; then
	echo "INFO [do_build_and_package_zcu.sh] Successfully packaged petalinux image"
else
	echo "ERROR [do_build_and_package_zcu.sh] Error packaging petalinux image"
	exit 1
fi

echo "INFO [do_build_and_package_zcu.sh] Zipping petalinux-sdimage.wic..."
gzip -f -k -v ./images/linux/petalinux-sdimage.wic
echo "INFO [do_build_and_package_zcu.sh] Finished"

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - UNIX_TIME_START))

# Calculate hours, minutes, and seconds
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))
echo "INFO [do_build_and_package_zcu.sh] Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s" 
