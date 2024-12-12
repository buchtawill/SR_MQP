#!/bin/bash
UNIX_TIME_START=$(date +%s)
echo "INFO [cross_compile_apps.sh] Sourcing petalinux settings"

if source /tools/Xilinx/PetaLinux/2023.1/settings.sh ; then
	echo "INFO [cross_compile_apps.sh] Successfully sourced petalinux settings"
else
	echo "ERROR [cross_compile_apps.sh] Error sourcing build tools"
	exit 1
fi

APPS_LIST=("v4l-to-fb0-dma v4l-to-fb0-aligned add-mult-test axi-dma-test dma-cpp print-fb-info vid-v4l-test dma-to-fb")
# APPS_LIST=("v4l-to-fb0-dma v4l-to-fb0-aligned")

for APP in ${APPS_LIST[@]}; do
    echo "INFO [cross_compile_apps.sh] Building $APP"
    if petalinux-build -c $APP ; then
        echo "INFO [cross_compile_apps.sh] Successfully built $APP"
    else
        echo "ERROR [cross_compile_apps.sh] Error building $APP"
        exit 1
    fi
done

echo "INFO [cross_compile_apps.sh] Rebuilding rootfs"
if petalinux-build -c rootfs ; then
    echo "INFO [cross_compile_apps.sh] Successfully built rootfs"
else
    echo "ERROR [cross_compile_apps.sh] Error building rootfs"
    exit 1
fi

# Extract the rootfs usr/bin folder
ROOTFS_ARCHIVE_PATH=./images/linux/rootfs.tar.gz
rm -rf ./remade_app_binaries
mkdir -p ./remade_app_binaries

for APP in ${APPS_LIST[@]}; do
    echo "INFO [cross_compile_apps.sh] Extracting $APP"
    tar -xzf $ROOTFS_ARCHIVE_PATH --strip-components=3 -C ./remade_app_binaries ./usr/bin/$APP
done

echo "INFO [cross_compile_apps.sh] SCPing new binaries to machine"
scp -r ./remade_app_binaries root@kv260-mqp-b.dyn.wpi.edu:/home/root

if [ $? -ne 0 ]; then
    echo "ERROR [cross_compile_apps.sh] Error SCPing new binaries to machine"
    exit 1
fi

echo "INFO [cross_compile_apps.sh] Success"

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - UNIX_TIME_START))

# Calculate hours, minutes, and seconds
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))
echo "INFO [cross_compile_apps.sh] Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s" 