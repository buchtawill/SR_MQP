#!/bin/python3.12

'''
This script will create vitis HLS projects with source and testbench files specified in 
hls_build_info.json. All source files for HLS projects will be kept in SR_MQP/HLS/src/<project_name>
'''

import os
import json
import time
import argparse


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

def create_hls_project(project_name:str, hls_build_info:dict, auto_overwrite: bool=False) -> int:
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
        file.write(f'open_project {project_name}\n')
        file.write(f'set_top {hls_build_info[project_name]['top_func']}\n')
        
        for src_name in hls_build_info[project_name]['src']:
            file.write(f'add_files src/{project_name}/{src_name}\n')
            
        for tb_name in hls_build_info[project_name]['tb']:
            file.write(f'add_files -tb src/{project_name}/{tb_name}\n')
            
        file.write('open_solution "solution1" -flow_target vivado\n')
        file.write('set_part {xck26-sfvc784-2LV-c}\n')
        file.write('create_clock -period 10 -name default\n')
        # file.write('csynth_design\n')
        # file.write('exit\n')

    # Create the tcl script to create the project
    with open(tmp_tcl_create, 'w') as file:
        file.write(f'open_tcl_project {tmp_tcl_build}\n')
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
        return result
    
    print(f"INFO [create_HLS_projects::create_hls_project] Project '{project_name}' created successfully.")
    os.remove(tmp_tcl_build)
    os.remove(tmp_tcl_create)
    
    return 0

if __name__ == '__main__':
    
    with open('./hls_build_info.json', 'r') as file:
        hls_build_info = json.load(file)

    parser = argparse.ArgumentParser(description='This script will checkout sources and create a Vitis HLS project(s)')
    parser.add_argument('--project', type=str, help='Name of the project')
    args = parser.parse_args()

    build_all = False
    if args.project:
        project = hls_build_info.get(args.project)
        if project:
            print(json.dumps(project, indent=4))
        else:
            print(f"ERROR [create_projects.py] Project '{args.project}' not found.")
    else:
        build_all = True
        
    if build_all:
        for project_name in hls_build_info:
            print(f"INFO [create_projects.py] Creating project '{project_name}'")
            result = create_hls_project(project_name, hls_build_info, auto_overwrite=True)
            if result != 0:
                print(f"ERROR [create_projects.py] Failed to create project {project_name}")
                exit(1)
    else:
        create_hls_project(args.project, hls_build_info)
        
    print("INFO [create_projects.py] All projects created successfully.")
    