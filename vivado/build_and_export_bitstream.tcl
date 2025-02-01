# Run this script from the vivado directory. NOT from the project directory.
# This script will open the project, validate the block design, run synthesis and implementation, and generate the bitstream.
# The hardware platform will be saved to the project directory.
# Run with the following command: vivado -mode batch -source example_script.tcl
# Author: Will Buchta Dec 2024

set project_name "kv260_vivado_project"
set project_dir "./$project_name"
set bitstream_dir "./bitstreams"

open_project "$project_dir/$project_name.xpr"

open_bd_design "$project_dir/$project_name.srcs/sources_1/bd/design_1/design_1.bd"
if {[catch {save_bd_design} result]} {
    puts "Error saving block design: $result"
    exit 1
}
if {[catch {validate_bd_design -force} result]} {
    puts "Error validating block design: $result"
    exit 1
}
if {[catch {save_bd_design} result]} {
    puts "Error saving block design: $result"
    exit 1
}

make_wrapper -files [get_files "$project_dir/$project_name.srcs/sources_1/bd/design_1/design_1.bd"] -top
update_compile_order -fileset sources_1

# Run synthesis
reset_run synth_1
launch_runs synth_1 -jobs 6
if {[catch {wait_on_run synth_1} result]} {
    puts "Error during synthesis: $result"
    exit 1
}

# Run implementation, generate the bitstream
reset_run impl_1
launch_runs impl_1 -to_step write_bitstream -jobs 6
if {[catch {wait_on_run impl_1} result]} {
    puts "Error during implementation: $result"
    exit 1
}

if {[catch {write_hw_platform -fixed -include_bit -force -file "$project_dir/kv260_upscaler.xsa"} result]} {
    puts "Error writing hardware platform: $result"
    exit 1
}

open_run impl_1

# Write the bitstream and export it to ./bitstreams
write_bitstream -force "$bitstream_dir/fpga_image.bit"

close_project
