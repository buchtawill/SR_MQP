#!/bin/bash

KV260_BOARD="kv260-b"

UNIX_TIME_START=$(date +%s)

function exit_with_fail {
    local message=$1
    echo "ERROR [build_project.sh] $message"
    echo "INFO [build_project.sh] Total execution time: $(($(date +%s) - UNIX_TIME_START)) seconds"

    exit 1
}

source /tools/Xilinx/Vivado/2023.1/settings64.sh

read -p "QUESTION [export_bitsream.sh] Do you want to SCP the new image to the board once complete? (y/n): " SCP_ANSWER

# Build the vivado project and export the bitstream
vivado -mode batch -source build_and_export_bitstream.tcl

# Check the exit status of the Vivado command
if [ $? -ne 0 ]; then
    # Send an email notification
    exit_with_fail "Vivado build failed"
else
    echo "INFO [build_project.sh] Vivado build completed successfully."
fi

bootgen -arch zynqmp -image ./bitstreams/image.bif -process_bitstream bin -w

# Check the exit status of the bootgen command
if [ $? -ne 0 ]; then
    # Send an email notification
    exit_with_fail "Bootgen failed"
else
    echo "INFO [build_project.sh] Bootgen completed successfully."
fi


if [[ "$SCP_ANSWER" == "y" ]]; then
    # SCP to the target KV260
    scp ./bitstreams/fpga_image.bit.bin petalinux@$KV260_BOARD.dyn.wpi.edu:/home/petalinux
    
    if [ $? -ne 0 ]; then
        # Send an email notification
        exit_with_fail "SCP failed"
    else
        echo "INFO [build_project.sh] SCP completed successfully."
    fi
fi  

TIME_AFTER_VIVADO=$(date +%s)
echo "INFO [export_bitstream.sh] Total execution time: $(($TIME_AFTER_VIVADO - UNIX_TIME_START)) seconds"
