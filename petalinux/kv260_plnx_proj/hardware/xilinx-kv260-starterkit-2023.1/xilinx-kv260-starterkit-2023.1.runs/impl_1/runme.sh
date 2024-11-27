#!/bin/sh

# 
# Vivado(TM)
# runme.sh: a Vivado-generated Runs Script for UNIX
# Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
# Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
# 

if [ -z "$PATH" ]; then
  PATH=/proj/xbuilds/SWIP/2023.1_0507_1903/installs/lin64/Vitis/2023.1/bin:/proj/xbuilds/SWIP/2023.1_0507_1903/installs/lin64/Vivado/2023.1/ids_lite/ISE/bin/lin64:/proj/xbuilds/SWIP/2023.1_0507_1903/installs/lin64/Vivado/2023.1/bin
else
  PATH=/proj/xbuilds/SWIP/2023.1_0507_1903/installs/lin64/Vitis/2023.1/bin:/proj/xbuilds/SWIP/2023.1_0507_1903/installs/lin64/Vivado/2023.1/ids_lite/ISE/bin/lin64:/proj/xbuilds/SWIP/2023.1_0507_1903/installs/lin64/Vivado/2023.1/bin:$PATH
fi
export PATH

if [ -z "$LD_LIBRARY_PATH" ]; then
  LD_LIBRARY_PATH=
else
  LD_LIBRARY_PATH=:$LD_LIBRARY_PATH
fi
export LD_LIBRARY_PATH

HD_PWD='/tmp/tmp.lkFs2pBRW2/temp/hw_design/k26_starter_kits/kv260/project/project_1.runs/impl_1'
cd "$HD_PWD"

HD_LOG=runme.log
/bin/touch $HD_LOG

ISEStep="./ISEWrap.sh"
EAStep()
{
     $ISEStep $HD_LOG "$@" >> $HD_LOG 2>&1
     if [ $? -ne 0 ]
     then
         exit
     fi
}

# pre-commands:
/bin/touch .init_design.begin.rst
EAStep vivado -log project_1_wrapper.vdi -applog -m64 -product Vivado -messageDb vivado.pb -mode batch -source project_1_wrapper.tcl -notrace


