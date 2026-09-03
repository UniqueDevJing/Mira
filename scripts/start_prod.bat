@echo off
REM 手动启动(可见窗口): 实时显示 uvicorn 日志与隧道状态, 关闭窗口即停止。
cd /d "C:\Users\Dominion\Desktop\rag-2.0"
"C:\Users\Dominion\Desktop\rag-2.0\venv\Scripts\python.exe" "C:\Users\Dominion\Desktop\rag-2.0\scripts\start_prod.py"
echo.
echo [已退出] 按任意键关闭窗口。
pause
