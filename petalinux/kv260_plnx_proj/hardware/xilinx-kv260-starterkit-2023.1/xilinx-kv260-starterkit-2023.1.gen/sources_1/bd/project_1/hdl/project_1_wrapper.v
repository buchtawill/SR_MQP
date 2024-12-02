//Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
//Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2023.1 (lin64) Build 3865809 Sun May  7 15:04:56 MDT 2023
//Date        : Sun May  7 23:14:08 2023
//Host        : xcosswbld12 running 64-bit Ubuntu 18.04.5 LTS
//Command     : generate_target project_1_wrapper.bd
//Design      : project_1_wrapper
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module project_1_wrapper
   (fan_en_b);
  output [0:0]fan_en_b;

  wire [0:0]fan_en_b;

  project_1 project_1_i
       (.fan_en_b(fan_en_b));
endmodule
