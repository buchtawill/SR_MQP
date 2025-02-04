@echo off

REM This script will either open the project or recreate it if it already exists

call "C:\Xilinx\Vivado\2023.1\settings64.bat"

if exist "kv260_vivado_project" (
    set /p choice="The directory 'kv260_vivado_project' already exists. Do you want to remove it? (y/n): "
    if /i "%choice%"=="y" (
        rmdir /s /q kv260_vivado_project
        echo Overwriting existing project
        vivado -mode batch -source create_project.tcl
    )
) else (
    vivado -mode batch -source create_project.tcl
)

if %errorlevel% neq 0 (
    echo Vivado project creation failed
    exit /b 1
)

vivado -mode batch -source start_gui.tcl
