@echo off
chcp 65001 >nul
echo ========================================
echo 博主智策 - 一键启动（实时数据更新）
echo ========================================
echo.

echo [提示] 请先配置环境变量
echo.
set /p DASHSCOPE_API_KEY="请输入阿里云 DashScope API Key: "
set /p DB_PASSWORD="请输入 MySQL 数据库密码: "

echo.
echo [1/3] 激活虚拟环境...
call .venv\Scripts\activate.bat

echo.
echo [2/3] 导出环境变量...
set DASHSCOPE_API_KEY=%DASHSCOPE_API_KEY%
set DB_PASSWORD=%DB_PASSWORD%

echo.
echo [3/3] 执行自动化启动流程...
python startup.py

pause
