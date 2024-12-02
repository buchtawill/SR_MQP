//Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
//Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2023.1 (lin64) Build 3865809 Sun May  7 15:04:56 MDT 2023
//Date        : Sun May  7 23:14:08 2023
//Host        : xcosswbld12 running 64-bit Ubuntu 18.04.5 LTS
//Command     : generate_target project_1.bd
//Design      : project_1
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

(* CORE_GENERATION_INFO = "project_1,IP_Integrator,{x_ipVendor=xilinx.com,x_ipLibrary=BlockDiagram,x_ipName=project_1,x_ipVersion=1.00.a,x_ipLanguage=VERILOG,numBlks=2,numReposBlks=2,numNonXlnxBlks=0,numHierBlks=0,maxHierDepth=0,numSysgenBlks=0,numHlsBlks=0,numHdlrefBlks=0,numPkgbdBlks=0,bdsource=USER,da_zynq_ultra_ps_e_cnt=1,synth_mode=OOC_per_IP}" *) (* HW_HANDOFF = "project_1.hwdef" *) 
module project_1
   (fan_en_b);
  output [0:0]fan_en_b;

  wire [0:0]xlslice_0_Dout;
  wire [2:0]zynq_ultra_ps_e_0_emio_ttc0_wave_o;

  assign fan_en_b[0] = xlslice_0_Dout;
  project_1_xlslice_0_0 xlslice_0
       (.Din(zynq_ultra_ps_e_0_emio_ttc0_wave_o),
        .Dout(xlslice_0_Dout));
  project_1_zynq_ultra_ps_e_0_0 zynq_ultra_ps_e_0
       (.emio_ttc0_wave_o(zynq_ultra_ps_e_0_emio_ttc0_wave_o),
        .pl_ps_irq0(1'b0));
endmodule
