
################################################################
# This is a generated script based on design: design_1
#
# Though there are limitations about the generated script,
# the main purpose of this utility is to make learning
# IP Integrator Tcl commands easier.
################################################################

namespace eval _tcl {
proc get_script_folder {} {
   set script_path [file normalize [info script]]
   set script_folder [file dirname $script_path]
   return $script_folder
}
}
variable script_folder
set script_folder [_tcl::get_script_folder]

################################################################
# Check if script is running in correct Vivado version.
################################################################
set scripts_vivado_version 2023.1
set current_vivado_version [version -short]

if { [string first $scripts_vivado_version $current_vivado_version] == -1 } {
   puts ""
   catch {common::send_gid_msg -ssname BD::TCL -id 2041 -severity "ERROR" "This script was generated using Vivado <$scripts_vivado_version> and is being run in <$current_vivado_version> of Vivado. Please run the script in Vivado <$scripts_vivado_version> then open the design in Vivado <$current_vivado_version>. Upgrade the design by running \"Tools => Report => Report IP Status...\", then run write_bd_tcl to create an updated script."}

   return 1
}

################################################################
# START
################################################################

# To test this script, run the following commands from Vivado Tcl console:
# source design_1_script.tcl


# The design that will be created by this Tcl script contains the following 
# module references:
# trim_output, tile_input_buffer

# Please add the sources of those modules before sourcing this Tcl script.

# If there is no project opened, this script will create a
# project, but make sure you do not have an existing project
# <./myproj/project_1.xpr> in the current working folder.

set list_projs [get_projects -quiet]
if { $list_projs eq "" } {
   create_project project_1 myproj -part xczu9eg-ffvb1156-2-e
   set_property BOARD_PART xilinx.com:zcu102:part0:3.4 [current_project]
}


# CHANGE DESIGN NAME HERE
variable design_name
set design_name design_1

# If you do not already have an existing IP Integrator design open,
# you can create a design using the following command:
#    create_bd_design $design_name

# Creating design if needed
set errMsg ""
set nRet 0

set cur_design [current_bd_design -quiet]
set list_cells [get_bd_cells -quiet]

if { ${design_name} eq "" } {
   # USE CASES:
   #    1) Design_name not set

   set errMsg "Please set the variable <design_name> to a non-empty value."
   set nRet 1

} elseif { ${cur_design} ne "" && ${list_cells} eq "" } {
   # USE CASES:
   #    2): Current design opened AND is empty AND names same.
   #    3): Current design opened AND is empty AND names diff; design_name NOT in project.
   #    4): Current design opened AND is empty AND names diff; design_name exists in project.

   if { $cur_design ne $design_name } {
      common::send_gid_msg -ssname BD::TCL -id 2001 -severity "INFO" "Changing value of <design_name> from <$design_name> to <$cur_design> since current design is empty."
      set design_name [get_property NAME $cur_design]
   }
   common::send_gid_msg -ssname BD::TCL -id 2002 -severity "INFO" "Constructing design in IPI design <$cur_design>..."

} elseif { ${cur_design} ne "" && $list_cells ne "" && $cur_design eq $design_name } {
   # USE CASES:
   #    5) Current design opened AND has components AND same names.

   set errMsg "Design <$design_name> already exists in your project, please set the variable <design_name> to another value."
   set nRet 1
} elseif { [get_files -quiet ${design_name}.bd] ne "" } {
   # USE CASES: 
   #    6) Current opened design, has components, but diff names, design_name exists in project.
   #    7) No opened design, design_name exists in project.

   set errMsg "Design <$design_name> already exists in your project, please set the variable <design_name> to another value."
   set nRet 2

} else {
   # USE CASES:
   #    8) No opened design, design_name not in project.
   #    9) Current opened design, has components, but diff names, design_name not in project.

   common::send_gid_msg -ssname BD::TCL -id 2003 -severity "INFO" "Currently there is no design <$design_name> in project, so creating one..."

   create_bd_design $design_name

   common::send_gid_msg -ssname BD::TCL -id 2004 -severity "INFO" "Making design <$design_name> as current_bd_design."
   current_bd_design $design_name

}

common::send_gid_msg -ssname BD::TCL -id 2005 -severity "INFO" "Currently the variable <design_name> is equal to \"$design_name\"."

if { $nRet != 0 } {
   catch {common::send_gid_msg -ssname BD::TCL -id 2006 -severity "ERROR" $errMsg}
   return $nRet
}

set bCheckIPsPassed 1
##################################################################
# CHECK IPs
##################################################################
set bCheckIPs 1
if { $bCheckIPs == 1 } {
   set list_check_ips "\ 
xilinx.com:ip:system_ila:1.1\
xilinx.com:ip:clk_wiz:6.0\
xilinx.com:ip:util_vector_logic:2.0\
xilinx.com:ip:proc_sys_reset:5.0\
xilinx.com:ip:xlconstant:1.1\
xilinx.com:ip:v_tc:6.2\
xilinx.com:ip:v_vid_in_axi4s:5.0\
"

   set list_ips_missing ""
   common::send_gid_msg -ssname BD::TCL -id 2011 -severity "INFO" "Checking if the following IPs exist in the project's IP catalog: $list_check_ips ."

   foreach ip_vlnv $list_check_ips {
      set ip_obj [get_ipdefs -all $ip_vlnv]
      if { $ip_obj eq "" } {
         lappend list_ips_missing $ip_vlnv
      }
   }

   if { $list_ips_missing ne "" } {
      catch {common::send_gid_msg -ssname BD::TCL -id 2012 -severity "ERROR" "The following IPs are not found in the IP Catalog:\n  $list_ips_missing\n\nResolution: Please add the repository containing the IP(s) to the project." }
      set bCheckIPsPassed 0
   }

}

##################################################################
# CHECK Modules
##################################################################
set bCheckModules 1
if { $bCheckModules == 1 } {
   set list_check_mods "\ 
trim_output\
tile_input_buffer\
"

   set list_mods_missing ""
   common::send_gid_msg -ssname BD::TCL -id 2020 -severity "INFO" "Checking if the following modules exist in the project's sources: $list_check_mods ."

   foreach mod_vlnv $list_check_mods {
      if { [can_resolve_reference $mod_vlnv] == 0 } {
         lappend list_mods_missing $mod_vlnv
      }
   }

   if { $list_mods_missing ne "" } {
      catch {common::send_gid_msg -ssname BD::TCL -id 2021 -severity "ERROR" "The following module(s) are not found in the project: $list_mods_missing" }
      common::send_gid_msg -ssname BD::TCL -id 2022 -severity "INFO" "Please add source files for the missing module(s) above."
      set bCheckIPsPassed 0
   }
}

if { $bCheckIPsPassed != 1 } {
  common::send_gid_msg -ssname BD::TCL -id 2023 -severity "WARNING" "Will not continue with creation of design due to the error(s) above."
  return 3
}

##################################################################
# DESIGN PROCs
##################################################################



# Procedure to create entire design; Provide argument to make
# procedure reusable. If parentCell is "", will use root.
proc create_root_design { parentCell } {

  variable script_folder
  variable design_name

  if { $parentCell eq "" } {
     set parentCell [get_bd_cells /]
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj


  # Create interface ports
  set user_si570_sysclk [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 user_si570_sysclk ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {300000000} \
   ] $user_si570_sysclk


  # Create ports
  set reset [ create_bd_port -dir I -type rst reset ]
  set_property -dict [ list \
   CONFIG.POLARITY {ACTIVE_HIGH} \
 ] $reset
  set LLC_0 [ create_bd_port -dir I LLC_0 ]
  set data_in_0 [ create_bd_port -dir I -from 7 -to 0 data_in_0 ]
  set HSYNC_0 [ create_bd_port -dir I HSYNC_0 ]
  set VSYNC_0 [ create_bd_port -dir I VSYNC_0 ]
  set FIELD_0 [ create_bd_port -dir I FIELD_0 ]

  # Create instance: system_ila_0, and set properties
  set system_ila_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:system_ila:1.1 system_ila_0 ]
  set_property -dict [list \
    CONFIG.C_DATA_DEPTH {131072} \
    CONFIG.C_MON_TYPE {MIX} \
    CONFIG.C_NUM_MONITOR_SLOTS {1} \
    CONFIG.C_NUM_OF_PROBES {18} \
    CONFIG.C_PROBE0_TYPE {0} \
    CONFIG.C_PROBE1_TYPE {0} \
    CONFIG.C_PROBE2_TYPE {0} \
    CONFIG.C_PROBE3_TYPE {0} \
    CONFIG.C_PROBE4_TYPE {0} \
    CONFIG.C_PROBE5_TYPE {0} \
    CONFIG.C_PROBE6_TYPE {0} \
    CONFIG.C_PROBE7_TYPE {0} \
    CONFIG.C_PROBE8_TYPE {0} \
    CONFIG.C_SLOT_0_INTF_TYPE {xilinx.com:interface:axis_rtl:1.0} \
  ] $system_ila_0


  # Create instance: clk_wiz_0, and set properties
  set clk_wiz_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz_0 ]
  set_property -dict [list \
    CONFIG.CLKIN1_JITTER_PS {33.330000000000005} \
    CONFIG.CLKOUT1_JITTER {101.475} \
    CONFIG.CLKOUT1_PHASE_ERROR {77.836} \
    CONFIG.CLK_IN1_BOARD_INTERFACE {user_si570_sysclk} \
    CONFIG.MMCM_CLKFBOUT_MULT_F {4.000} \
    CONFIG.MMCM_CLKIN1_PERIOD {3.333} \
    CONFIG.MMCM_CLKIN2_PERIOD {10.0} \
    CONFIG.OPTIMIZE_CLOCKING_STRUCTURE_EN {true} \
    CONFIG.PRIM_SOURCE {Differential_clock_capable_pin} \
    CONFIG.RESET_BOARD_INTERFACE {reset} \
    CONFIG.RESET_PORT {resetn} \
    CONFIG.RESET_TYPE {ACTIVE_LOW} \
    CONFIG.USE_BOARD_FLOW {true} \
  ] $clk_wiz_0


  # Create instance: reset_inv_0, and set properties
  set reset_inv_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:2.0 reset_inv_0 ]
  set_property -dict [list \
    CONFIG.C_OPERATION {not} \
    CONFIG.C_SIZE {1} \
  ] $reset_inv_0


  # Create instance: rst_clk_wiz_0_100M, and set properties
  set rst_clk_wiz_0_100M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_clk_wiz_0_100M ]

  # Create instance: xlconstant_0, and set properties
  set xlconstant_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 xlconstant_0 ]

  # Create instance: v_tc_0, and set properties
  set v_tc_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:v_tc:6.2 v_tc_0 ]
  set_property -dict [list \
    CONFIG.DET_FIELDID_EN {true} \
    CONFIG.GEN_FIELDID_EN {true} \
    CONFIG.HAS_AXI4_LITE {false} \
    CONFIG.HAS_INTC_IF {false} \
    CONFIG.INTERLACE_EN {true} \
    CONFIG.SYNC_EN {false} \
    CONFIG.VIDEO_MODE {576i} \
    CONFIG.active_chroma_generation {false} \
    CONFIG.active_video_detection {false} \
    CONFIG.horizontal_blank_detection {true} \
    CONFIG.horizontal_sync_generation {true} \
    CONFIG.vertical_blank_detection {true} \
    CONFIG.vertical_sync_generation {true} \
  ] $v_tc_0


  # Create instance: v_vid_in_axi4s_0, and set properties
  set v_vid_in_axi4s_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:v_vid_in_axi4s:5.0 v_vid_in_axi4s_0 ]
  set_property -dict [list \
    CONFIG.C_ADDR_WIDTH {12} \
    CONFIG.C_HAS_ASYNC_CLK {1} \
    CONFIG.C_M_AXIS_VIDEO_FORMAT {8} \
  ] $v_vid_in_axi4s_0


  # Create instance: trim_output_0, and set properties
  set block_name trim_output
  set block_cell_name trim_output_0
  if { [catch {set trim_output_0 [create_bd_cell -type module -reference $block_name $block_cell_name] } errmsg] } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2095 -severity "ERROR" "Unable to add referenced block <$block_name>. Please add the files for ${block_name}'s definition into the project."}
     return 1
   } elseif { $trim_output_0 eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2096 -severity "ERROR" "Unable to referenced block <$block_name>. Please add the files for ${block_name}'s definition into the project."}
     return 1
   }
  
  # Create instance: tile_input_buffer_0, and set properties
  set block_name tile_input_buffer
  set block_cell_name tile_input_buffer_0
  if { [catch {set tile_input_buffer_0 [create_bd_cell -type module -reference $block_name $block_cell_name] } errmsg] } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2095 -severity "ERROR" "Unable to add referenced block <$block_name>. Please add the files for ${block_name}'s definition into the project."}
     return 1
   } elseif { $tile_input_buffer_0 eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2096 -severity "ERROR" "Unable to referenced block <$block_name>. Please add the files for ${block_name}'s definition into the project."}
     return 1
   }
  
  # Create interface connections
  connect_bd_intf_net -intf_net user_si570_sysclk_1 [get_bd_intf_ports user_si570_sysclk] [get_bd_intf_pins clk_wiz_0/CLK_IN1_D]

  # Create port connections
  connect_bd_net -net FIELD_0_1 [get_bd_ports FIELD_0] [get_bd_pins system_ila_0/probe5] [get_bd_pins v_tc_0/field_id_in]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets FIELD_0_1]
  connect_bd_net -net HSYNC_0_1 [get_bd_ports HSYNC_0] [get_bd_pins system_ila_0/probe6] [get_bd_pins v_tc_0/hsync_in]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets HSYNC_0_1]
  connect_bd_net -net LLC_0_1 [get_bd_ports LLC_0] [get_bd_pins system_ila_0/probe7] [get_bd_pins v_vid_in_axi4s_0/vid_io_in_clk] [get_bd_pins v_tc_0/clk]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets LLC_0_1]
  connect_bd_net -net VSYNC_0_1 [get_bd_ports VSYNC_0] [get_bd_pins system_ila_0/probe8] [get_bd_pins v_tc_0/vsync_in]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets VSYNC_0_1]
  connect_bd_net -net clk_wiz_0_clk_out1 [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins system_ila_0/clk] [get_bd_pins rst_clk_wiz_0_100M/slowest_sync_clk] [get_bd_pins v_vid_in_axi4s_0/aclk] [get_bd_pins trim_output_0/aclk] [get_bd_pins tile_input_buffer_0/aclk]
  connect_bd_net -net clk_wiz_0_locked [get_bd_pins clk_wiz_0/locked] [get_bd_pins rst_clk_wiz_0_100M/dcm_locked]
  connect_bd_net -net data_in_0_1 [get_bd_ports data_in_0] [get_bd_pins system_ila_0/probe4] [get_bd_pins tile_input_buffer_0/data_in]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets data_in_0_1]
  connect_bd_net -net reset_1 [get_bd_ports reset] [get_bd_pins reset_inv_0/Op1] [get_bd_pins v_vid_in_axi4s_0/vid_io_in_reset]
  connect_bd_net -net reset_inv_0_Res [get_bd_pins reset_inv_0/Res] [get_bd_pins clk_wiz_0/resetn] [get_bd_pins rst_clk_wiz_0_100M/ext_reset_in] [get_bd_pins system_ila_0/probe10] [get_bd_pins v_vid_in_axi4s_0/aresetn] [get_bd_pins trim_output_0/aresetn] [get_bd_pins v_tc_0/resetn] [get_bd_pins tile_input_buffer_0/aresetn] [get_bd_pins system_ila_0/resetn]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets reset_inv_0_Res]
  connect_bd_net -net tile_input_buffer_0_m_axis_tdata [get_bd_pins tile_input_buffer_0/m_axis_tdata] [get_bd_pins v_vid_in_axi4s_0/vid_data] [get_bd_pins system_ila_0/probe11]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets tile_input_buffer_0_m_axis_tdata]
  connect_bd_net -net trim_output_0_m_axis_tdata [get_bd_pins trim_output_0/m_axis_tdata] [get_bd_pins system_ila_0/probe0]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets trim_output_0_m_axis_tdata]
  connect_bd_net -net trim_output_0_m_axis_tlast [get_bd_pins trim_output_0/m_axis_tlast] [get_bd_pins system_ila_0/probe1]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets trim_output_0_m_axis_tlast]
  connect_bd_net -net trim_output_0_m_axis_tuser [get_bd_pins trim_output_0/m_axis_tuser] [get_bd_pins system_ila_0/probe2]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets trim_output_0_m_axis_tuser]
  connect_bd_net -net trim_output_0_m_axis_tvalid [get_bd_pins trim_output_0/m_axis_tvalid] [get_bd_pins system_ila_0/probe3]
  connect_bd_net -net trim_output_0_s_axis_tready [get_bd_pins trim_output_0/s_axis_tready] [get_bd_pins v_vid_in_axi4s_0/m_axis_video_tready]
  connect_bd_net -net v_tc_0_active_video_out [get_bd_pins v_tc_0/active_video_out] [get_bd_pins v_vid_in_axi4s_0/vid_active_video] [get_bd_pins system_ila_0/probe12]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets v_tc_0_active_video_out]
  connect_bd_net -net v_tc_0_field_id_out [get_bd_pins v_tc_0/field_id_out] [get_bd_pins v_vid_in_axi4s_0/vid_field_id] [get_bd_pins system_ila_0/probe15]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets v_tc_0_field_id_out]
  connect_bd_net -net v_tc_0_hblank_out [get_bd_pins v_tc_0/hblank_out] [get_bd_pins v_vid_in_axi4s_0/vid_hblank] [get_bd_pins system_ila_0/probe13]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets v_tc_0_hblank_out]
  connect_bd_net -net v_tc_0_hsync_out [get_bd_pins v_tc_0/hsync_out] [get_bd_pins v_vid_in_axi4s_0/vid_hsync] [get_bd_pins system_ila_0/probe17]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets v_tc_0_hsync_out]
  connect_bd_net -net v_tc_0_vblank_out [get_bd_pins v_tc_0/vblank_out] [get_bd_pins v_vid_in_axi4s_0/vid_vblank] [get_bd_pins system_ila_0/probe14]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets v_tc_0_vblank_out]
  connect_bd_net -net v_tc_0_vsync_out [get_bd_pins v_tc_0/vsync_out] [get_bd_pins v_vid_in_axi4s_0/vid_vsync] [get_bd_pins system_ila_0/probe16]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets v_tc_0_vsync_out]
  connect_bd_net -net v_vid_in_axi4s_0_m_axis_video_tdata [get_bd_pins v_vid_in_axi4s_0/m_axis_video_tdata] [get_bd_pins trim_output_0/s_axis_tdata]
  connect_bd_net -net v_vid_in_axi4s_0_m_axis_video_tlast [get_bd_pins v_vid_in_axi4s_0/m_axis_video_tlast] [get_bd_pins trim_output_0/s_axis_tlast]
  connect_bd_net -net v_vid_in_axi4s_0_m_axis_video_tuser [get_bd_pins v_vid_in_axi4s_0/m_axis_video_tuser] [get_bd_pins trim_output_0/s_axis_tuser]
  connect_bd_net -net v_vid_in_axi4s_0_m_axis_video_tvalid [get_bd_pins v_vid_in_axi4s_0/m_axis_video_tvalid] [get_bd_pins trim_output_0/s_axis_tvalid]
  connect_bd_net -net xlconstant_0_dout [get_bd_pins xlconstant_0/dout] [get_bd_pins system_ila_0/probe9] [get_bd_pins v_vid_in_axi4s_0/vid_io_in_ce] [get_bd_pins v_vid_in_axi4s_0/aclken] [get_bd_pins trim_output_0/m_axis_tready] [get_bd_pins v_tc_0/clken] [get_bd_pins v_tc_0/det_clken] [get_bd_pins v_tc_0/gen_clken] [get_bd_pins v_vid_in_axi4s_0/axis_enable]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_nets xlconstant_0_dout]

  # Create address segments


  # Restore current instance
  current_bd_instance $oldCurInst

  validate_bd_design
  save_bd_design
}
# End of create_root_design()


##################################################################
# MAIN FLOW
##################################################################

create_root_design ""


