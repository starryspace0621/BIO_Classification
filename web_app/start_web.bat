@echo off
echo Starting Medical Image Classification Web Application...

:: 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.6+ and try again
    pause
    exit /b 1
)

:: 检查并安装必要的依赖
echo Installing required packages...
pip install flask flask-socketio >nul 2>&1
if errorlevel 1 (
    echo Error: Failed to install required packages
    pause
    exit /b 1
)

:: 创建必要的目录
if not exist "static\uploads" mkdir "static\uploads"
if not exist "static\images" mkdir "static\images"

:: 启动web应用
echo Starting web server...
echo Please wait...
echo.
echo Once the server starts, open your web browser and go to:
echo http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py

pause 