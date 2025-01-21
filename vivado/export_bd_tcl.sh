#!/bin/bash

KV260_BOARD="kv260b"

UNIX_TIME_START=$(date +%s)

function exit_with_fail {
    local message=$1
    echo "ERROR [build_project.sh] $message"
    echo "INFO [build_project.sh] Total execution time: $(($(date +%s) - UNIX_TIME_START)) seconds"

    exit 1
}

source /tools/Xilinx/Vivado/2023.1/settings64.sh

# Build the vivado project and export the bitstream
vivado -mode batch -source export_bd.tcl

# Check the exit status of the Vivado command
if [ $? -ne 0 ]; then
    # Send an email notification
    exit_with_fail "Vivado build failed"
else
    echo "INFO [build_project.sh] Successfully exported the block design."
fi