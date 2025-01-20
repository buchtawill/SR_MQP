#!/bin/bash
UNIX_TIME_START=$(date +%s)
echo "INFO [cross_compile_apps.sh] Sourcing petalinux settings"

if source /tools/Xilinx/PetaLinux/2023.1/settings.sh ; then
	echo "INFO [cross_compile_apps.sh] Successfully sourced petalinux settings"
else
	echo "ERROR [cross_compile_apps.sh] Error sourcing build tools"
	exit 1
fi

# APPS_LIST=("v4l-to-fb0-dma v4l-to-fb0-aligned add-mult-test print-fb-info")
APPS_LIST=("interpolate2x")

# App location: 
# kv260_plnx_proj/build/tmp/work/cortexa72-cortexa53-xilinx-linux/<app name>/1.0-r0/<app name>
# kv260_plnx_proj/build/tmp/work/cortexa72-cortexa53-xilinx-linux/v4l-to-fb0-aligned/1.0-r0/v4l-to-fb0-aligned

PROJECT_DIR=$(pwd)

for APP in ${APPS_LIST[@]}; do
    echo "INFO [cross_compile_apps.sh] Building $APP"

    # Build the app, leaving the build directory
    if petalinux-build -c $APP -x compile; then
        echo "INFO [cross_compile_apps.sh] Successfully built $APP"
    else
        echo "ERROR [cross_compile_apps.sh] Error building $APP"
        exit 1
    fi
done


# Old method: rebuild rootfs and extract the usr/bin folder
# ROOTFS_ARCHIVE_PATH=./images/linux/rootfs.tar.gz
# New method: Copy app binaries directly

rm -rf ./remade_app_binaries
mkdir -p ./remade_app_binaries
for APP in ${APPS_LIST[@]}; do
    echo "INFO [cross_compile_apps.sh] Copying $APP"
    BUILD_PATH=$PROJECT_DIR/build/tmp/work/cortexa72-cortexa53-xilinx-linux/$APP/1.0-r0/$APP
    cp $BUILD_PATH ./remade_app_binaries
done

# echo "INFO [cross_compile_apps.sh] SCPing new binaries to machine"
# scp -r ./remade_app_binaries root@kv260-mqp-b.dyn.wpi.edu:/home/root

# if [ $? -ne 0 ]; then
#     echo "ERROR [cross_compile_apps.sh] Error SCPing new binaries to machine"
#     exit 1
# fi

echo "INFO [cross_compile_apps.sh] Success"

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - UNIX_TIME_START))

# Calculate hours, minutes, and seconds
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))
echo "INFO [cross_compile_apps.sh] Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s" 
