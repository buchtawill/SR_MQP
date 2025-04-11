@echo off

REM This script will either open the project or recreate it if it already exists

call "C:\Xilinx\Vivado\2023.1\settings64.bat"

if exist "zcu102_vivado_project" (
    set /p choice="The directory 'zcu102_vivado_project' already exists. Do you want to remove it? (overwrite/n): "
    if /i "%choice%"=="overwrite" (
        rmdir /s /q zcu102_vivado_project
        echo Overwriting existing project
        vivado -mode batch -source create_project_zcu102.tcl
    )
) else (
    vivado -mode batch -source create_project_zcu102.tcl
)

if %errorlevel% neq 0 (
    echo Vivado project creation failed
    exit /b 1
)

vivado -mode batch -source start_gui_zcu.tcl
