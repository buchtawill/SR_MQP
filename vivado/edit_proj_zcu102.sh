#! /bin/bash

# This script will either open the project or recreate it if it already exists

source /tools/Xilinx/Vivado/2023.1/settings64.sh

if [ -d "zcu102_vivado_project" ]; then
    read -p "The directory 'zcu102_vivado_project' already exists. Do you want to remove it? (overwrite/n): " choice
    if [ "$choice" = "overwrite" ]; then
        rm -rf zcu102_vivado_project
        echo "Overwriting existing project"
        vivado -mode batch -source create_project_zcu102.tcl
    fi
# the project directory does not exist
else 
    vivado -mode batch -source create_project_zcu102.tcl
fi

if [ $? -ne 0 ]; then
    echo "Vivado project creation failed"
    exit 1
fi

vivado -mode batch -source start_gui_zcu.tcl