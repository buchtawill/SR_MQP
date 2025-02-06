#!/bin/python3.12

'''
This script will create vitis HLS projects with source and testbench files specified in 
hls_build_info.json. All source files for HLS projects will be kept in SR_MQP/HLS/src/<project_name>
'''

import os
import json
import time
import argparse
import threading

'''
Example build_info['bilinear_interpolation']:
{
    "src": [
        "bilinear_interpolation.cpp",
        "bilinear_interpolation.h",
        "../image_coin_tile.h"
    ],
    "tb": [
        "bilinear_interpolation_tb.cpp"
    ],
    "top_func": "bilinear_interpolation"
}
'''

def create_hls_project(project_name:str, hls_build_info:dict, auto_overwrite: bool=False, export_ip:bool=False) -> int:
    """
    Create a Vitis HLS project for the given project name. 
    Under the hood, it creates two temporary tcl files for building and creating the project.
    
    Returns:
        0: Success
        1: Error
    """
    
    # Check if the project exists
    if os.path.exists(project_name):
        if(auto_overwrite):
            print(f"INFO [create_projects::create_hls_project] Removing existing directory '{project_name}'")
            os.system(f'rm -rf {project_name}')
        else:
            response = input(f"The directory '{project_name}' already exists. Do you want to remove it? (y/n): ")
            if response.lower() == 'y':
                os.system(f'rm -rf {project_name}')
            else:
                print(f"ERROR [create_projects::create_hls_project] Directory '{project_name}' already exists.")
                return 1
    
    # Create custom names
    temp_time = int(time.time())
    tmp_tcl_build = f'build_{project_name}_{temp_time}.tcl'
    tmp_tcl_create = f'create_{project_name}_{temp_time}.tcl'
    
    # Create the tcl script to make the project and import sources
    with open(tmp_tcl_build, 'w') as file:
        file.write(f'open_project {project_name}_proj\n')
        file.write(f"set_top {hls_build_info[project_name]['top_func']}\n")
        
        for src_name in hls_build_info[project_name]['src']:
            file.write(f'add_files ../src/{project_name}/{src_name}\n')
            
        for tb_name in hls_build_info[project_name]['tb']:
            file.write(f'add_files -tb ../src/{project_name}/{tb_name}\n')
            
        file.write('open_solution "solution1" -flow_target vivado\n')
        file.write('set_part {xck26-sfvc784-2LV-c}\n')
        file.write('create_clock -period 10 -name default\n')

    # Create the tcl script to create the project
    with open(tmp_tcl_create, 'w') as file:
        file.write(f'open_tcl_project {tmp_tcl_build}\n')
        
        if export_ip:
            file.write('csynth_design\n')
            file.write(f'export_design -format ip_catalog -flow impl -ipname {project_name}\n')
            
        file.write('exit\n')

    # Check the OS type and run the appropriate script
    if os.name == 'posix':
        result = os.system(f'./help_create_HLS_projects.sh {tmp_tcl_create}')
    elif os.name == 'nt':
        result = os.system(f'help_create_HLS_projects.bat {tmp_tcl_create}')
    else:
        print(f"ERROR [create_HLS_projects::create_hls_project] Unsupported OS: {os.name}")
        return 1
    
    if result != 0:
        print(f"ERROR [create_HLS_projects::create_hls_project] Vitis HLS project creation failed with exit code {result}")
        exit(result)
        return result
    
    print(f"INFO [create_HLS_projects::create_hls_project] Project '{project_name}' created successfully.")
    os.remove(tmp_tcl_build)
    os.remove(tmp_tcl_create)
    
    return 0

def create_project_thread(project_name, hls_build_info, export_ip):
    result = create_hls_project(project_name, hls_build_info, auto_overwrite=True, export_ip=export_ip)
    if result != 0:
        print(f"ERROR [create_projects.py] Failed to create project {project_name}")
    return result

if __name__ == '__main__':
    
    with open('./hls_build_info.json', 'r') as file:
        hls_build_info = json.load(file)

    parser = argparse.ArgumentParser(description='This script will checkout sources and create a Vitis HLS project(s)')
    parser.add_argument('--project', type=str, help='Name of the project')
    parser.add_argument('--export_ip', action='store_true', help='Flag to export IP after project creation')
    parser.add_argument('--clean', action='store_true', help='If this flag is set, delete all existing projects in the build directory')
    args = parser.parse_args()
    
    if(args.clean):
        for project_name in hls_build_info:
            proj_path = f'{project_name}_proj'
            if os.path.exists(proj_path):
                if os.name == 'posix':
                    os.system(f'rm -rf {proj_path}')
                elif os.name == 'nt':
                    os.system(f'rmdir /s /q {proj_path}')
        exit(0)

    build_all = False
    fail = False
    if args.project:
        project = hls_build_info.get(args.project)
        if project:
            create_hls_project(args.project, hls_build_info, export_ip=args.export_ip)
        else:
            print(f"ERROR [create_projects.py] Project '{args.project}' not found.")
    else:
        threads = []

        for project_name in hls_build_info:
            print(f"INFO [create_projects.py] Creating project '{project_name}'")
            thread = threading.Thread(target=create_project_thread, args=(project_name,hls_build_info,args.export_ip))
            threads.append(thread)
            thread.start()
