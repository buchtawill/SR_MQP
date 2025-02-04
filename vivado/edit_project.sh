#! /bin/bash

# This script will either open the project or recreate it if it already exists

source /tools/Xilinx/Vivado/2023.1/settings64.sh

if [ -d "kv260_vivado_project" ]; then
    read -p "The directory 'kv260_vivado_project' already exists. Do you want to remove it? (overwrite/n): " choice
    if [ "$choice" = "overwrite" ]; then
        rm -rf kv260_vivado_project
        echo "Overwriting existing project"
        vivado -mode batch -source create_project.tcl
    fi
# the project directory does not exist
else 
    vivado -mode batch -source create_project.tcl
fi

if [ $? -ne 0 ]; then
    echo "Vivado project creation failed"
    exit 1
fi

vivado -mode batch -source start_gui.tcl
