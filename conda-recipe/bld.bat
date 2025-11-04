@echo off

REM Install pip dependencies that aren't available in conda
pip install cellpose>=2.0.0 ^
    stardist>=0.8.0 ^
    opencv-rolling-ball>=1.0.1 ^
    xlsxwriter>=3.0.0 ^
    gradio_client>=1.0.0

REM Install the package
%PYTHON% -m pip install . --no-deps --ignore-installed -vv
if errorlevel 1 exit 1

