#!/bin/bash

MAIN_DIR=$(pwd)
PLNX_PROJ_DIR=$MAIN_DIR/petalinux/kv260_plnx_proj
VIVADO_PROJ_DIR=$MAIN_DIR/vivado/kv260_vivado_proj
XSA_FILE_PATH=$VIVADO_PROJ_DIR/kv260_upscaler.xsa

if rm ./build.log; then
    # This is a workaround to avoid the error message when the file does not exist
    # Successfully removed build.log
    touch ./build.log
else
    touch ./build.log
fi


now=$(date)
echo "INFO [build_project.sh] Starting execution on $now" >> $MAIN_DIR/build.log
UNIX_TIME_START=$(date +%s)

echo "INFO [build_project.sh] Changing to vivado dir and running build script" >> $MAIN_DIR/build.log
cd ./vivado

source /tools/Xilinx/Vivado/2023.1/settings64.sh

# Run Vivado in batch mode
vivado -mode batch -source build_proj.tcl >> $MAIN_DIR/build.log

# Check the exit status of the Vivado command
if [ $? -ne 0 ]; then
    echo "ERROR [build_project.sh] Vivado build failed. Exiting." >> $MAIN_DIR/build.log
    echo "INFO [build_project.sh] Total execution time: $(($(date +%s) - UNIX_TIME_START)) seconds" >> $MAIN_DIR/build.log
    exit 1
else
    echo "INFO [build_project.sh] Vivado build completed successfully." >> $MAIN_DIR/build.log
fi

TIME_AFTER_VIVADO=$(date +%s)

###########################################################################
# Change to petalinux project directory and import hardware configuration #
###########################################################################
echo "INFO [build_project.sh] Changing to petalinux dir and sourcing settings" >> $MAIN_DIR/build.log
cd $PLNX_PROJ_DIR

if source /tools/Xilinx/PetaLinux/2023.1/settings.sh ; then
	echo "INFO [build_and_package.sh] Successfully sourced petalinux settings" >> $MAIN_DIR/build.log
else
	echo "ERROR [build_and_package.sh] Error sourcing build tools" >> $MAIN_DIR/build.log
    echo "INFO [build_project.sh] Total execution time: $(($(date +%s) - UNIX_TIME_START)) seconds" >> $MAIN_DIR/build.log
	exit 1
fi

echo "INFO [build_project.sh] Importing hardware configuration" >> $MAIN_DIR/build.log
petalinux-config --get-hw-description=$XSA_FILE_PATH --silentconfig >> $MAIN_DIR/build.log

# Check the exit status of the petalinux-config command
if [ $? -ne 0 ]; then
    echo "ERROR [build_project.sh] Petalinux configuration failed. Exiting." >> $MAIN_DIR/build.log
    echo "INFO [build_project.sh] Total execution time: $(($(date +%s) - UNIX_TIME_START)) seconds" >> $MAIN_DIR/build.log
    exit 1
else
    echo "INFO [build_project.sh] Petalinux configuration completed successfully." >> $MAIN_DIR/build.log
fi

TIME_AFTER_PETALINUX_CONFIG=$(date +%s)

# Run the build script
echo "INFO [build_project.sh] Running petalinux build script" >> $MAIN_DIR/build.log
./do_build_and_package.sh >> $MAIN_DIR/build.log

now=$(date)
echo "INFO [build_project.sh] Finished execution at $now"

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - UNIX_TIME_START))

# Calculate hours, minutes, and seconds
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

# Print the formatted execution time
VIVADO_ELAPSED_TIME=$((TIME_AFTER_VIVADO - UNIX_TIME_START))
VIVADO_HOURS=$((VIVADO_ELAPSED_TIME / 3600))
VIVADO_MINUTES=$(((VIVADO_ELAPSED_TIME % 3600) / 60))
VIVADO_SECONDS=$((VIVADO_ELAPSED_TIME % 60))

# Calculate Petalinux configuration time
PETALINUX_CONFIG_ELAPSED_TIME=$((TIME_AFTER_PETALINUX_CONFIG - TIME_AFTER_VIVADO))
PETALINUX_CONFIG_HOURS=$((PETALINUX_CONFIG_ELAPSED_TIME / 3600))
PETALINUX_CONFIG_MINUTES=$(((PETALINUX_CONFIG_ELAPSED_TIME % 3600) / 60))
PETALINUX_CONFIG_SECONDS=$((PETALINUX_CONFIG_ELAPSED_TIME % 60))

# Calculate Petalinux build time
PETALINUX_BUILD_ELAPSED_TIME=$((END_TIME - TIME_AFTER_PETALINUX_CONFIG))
PETALINUX_BUILD_HOURS=$((PETALINUX_BUILD_ELAPSED_TIME / 3600))
PETALINUX_BUILD_MINUTES=$(((PETALINUX_BUILD_ELAPSED_TIME % 3600) / 60))
PETALINUX_BUILD_SECONDS=$((PETALINUX_BUILD_ELAPSED_TIME % 60))

echo "INFO [build_project.sh] Vivado build time:     ${VIVADO_HOURS}h ${VIVADO_MINUTES}m ${VIVADO_SECONDS}s" >> $MAIN_DIR/build.log
echo "INFO [build_project.sh] Vivado build time:     ${VIVADO_HOURS}h ${VIVADO_MINUTES}m ${VIVADO_SECONDS}s"
echo "INFO [build_project.sh] Petalinux config time: ${PETALINUX_CONFIG_HOURS}h ${PETALINUX_CONFIG_MINUTES}m ${PETALINUX_CONFIG_SECONDS}s" >> $MAIN_DIR/build.log
echo "INFO [build_project.sh] Petalinux config time: ${PETALINUX_CONFIG_HOURS}h ${PETALINUX_CONFIG_MINUTES}m ${PETALINUX_CONFIG_SECONDS}s"
echo "INFO [build_project.sh] Petalinux build time:  ${PETALINUX_BUILD_HOURS}h ${PETALINUX_BUILD_MINUTES}m ${PETALINUX_BUILD_SECONDS}s" >> $MAIN_DIR/build.log
echo "INFO [build_project.sh] Petalinux build time:  ${PETALINUX_BUILD_HOURS}h ${PETALINUX_BUILD_MINUTES}m ${PETALINUX_BUILD_SECONDS}s"
echo "INFO [build_project.sh] Total execution time:  ${HOURS}h ${MINUTES}m ${SECONDS}s" >> $MAIN_DIR/build.log
echo "INFO [build_project.sh] Total execution time:  ${HOURS}h ${MINUTES}m ${SECONDS}s" 