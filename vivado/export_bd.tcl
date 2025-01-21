
set project_name "kv260_vivado_proj"
set project_dir "./$project_name"

open_project "$project_dir/$project_name.xpr"

open_bd_design "$project_dir/$project_name.srcs/sources_1/bd/design_1/design_1.bd"

if {[catch {save_bd_design} result]} {
    puts "Error saving block design: $result"
    exit 1
}
