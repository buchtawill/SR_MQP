-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
-- --------------------------------------------------------------------------------
-- Tool Version: Vivado v.2023.1 (lin64) Build 3865809 Sun May  7 15:04:56 MDT 2023
-- Date        : Sun May  7 23:15:40 2023
-- Host        : xcosswbld12 running 64-bit Ubuntu 18.04.5 LTS
-- Command     : write_vhdl -force -mode synth_stub
--               /tmp/tmp.lkFs2pBRW2/temp/hw_design/k26_starter_kits/kv260/project/project_1.gen/sources_1/bd/project_1/ip/project_1_zynq_ultra_ps_e_0_0/project_1_zynq_ultra_ps_e_0_0_stub.vhdl
-- Design      : project_1_zynq_ultra_ps_e_0_0
-- Purpose     : Stub declaration of top-level module interface
-- Device      : xck26-sfvc784-2LV-c
-- --------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity project_1_zynq_ultra_ps_e_0_0 is
  Port ( 
    emio_ttc0_wave_o : out STD_LOGIC_VECTOR ( 2 downto 0 );
    pl_ps_irq0 : in STD_LOGIC_VECTOR ( 0 to 0 );
    pl_resetn0 : out STD_LOGIC;
    pl_clk0 : out STD_LOGIC;
    pl_clk1 : out STD_LOGIC
  );

end project_1_zynq_ultra_ps_e_0_0;

architecture stub of project_1_zynq_ultra_ps_e_0_0 is
attribute syn_black_box : boolean;
attribute black_box_pad_pin : string;
attribute syn_black_box of stub : architecture is true;
attribute black_box_pad_pin of stub : architecture is "emio_ttc0_wave_o[2:0],pl_ps_irq0[0:0],pl_resetn0,pl_clk0,pl_clk1";
attribute X_CORE_INFO : string;
attribute X_CORE_INFO of stub : architecture is "zynq_ultra_ps_e_v3_5_0_zynq_ultra_ps_e,Vivado 2023.1";
begin
end;
